"""
宠物个体识别 - 模型模块（阶段 1：单分支正脸 baseline）

结构：timm EfficientNet-B0 (ImageNet 预训练) → embedding head (512维) → ArcFace 分类头

- backbone 特征 → 一层 FC 压到 512 维 embedding
- 训练时 embedding 经 ArcFace head 算 margin-softmax 损失
- 推理时【只取 512 维 embedding】，两张图算余弦相似度判同异

ArcFace 使用 pytorch-metric-learning 的实现（数值稳定，避免自写踩 NaN）。

"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pytorch_metric_learning.losses import ArcFaceLoss


def _build_backbone(backbone_name, pretrained, weights_file=None):
    """
    创建 timm backbone，去掉分类头只输出 pooled 特征向量。

    num_classes=0 时 timm 会移除分类层，forward 直接返回全局池化后的特征，
    特征维度可由 backbone.num_features 读取。

    :param backbone_name: timm 模型名，如 efficientnet_b0
    :param pretrained: 是否加载预训练权重
    :param weights_file: 本地权重文件路径（.safetensors / .pth）。
        指定后经 pretrained_cfg_overlay 从本地文件加载，训练全程不联网；
        为 None 时走 timm 默认逻辑（从 HuggingFace 下载到 ~/.cache/huggingface）。
        仅在 pretrained=True 时生效。
    :return: timm 模型实例
    """
    kwargs = {}
    if weights_file:
        # 提前校验文件存在：路径写错时 timm 的报错信息很晦涩，这里直接失败更好定位
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"本地预训练权重不存在: {weights_file}")
        kwargs["pretrained_cfg_overlay"] = dict(file=weights_file)
    return timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, **kwargs)


class PetEmbeddingNet(nn.Module):
    """
    单分支特征提取网络：图片 → 512 维 L2 归一化 embedding。
    """

    def __init__(self, backbone_name="efficientnet_b0", embedding_dim=512, pretrained=True,
                 weights_file=None):
        """
        :param backbone_name: timm backbone 名（默认 EfficientNet-B0）
        :param embedding_dim: 输出 embedding 维度
        :param pretrained: 是否加载 ImageNet 预训练权重（pretrained=True 自动下载）
        :param weights_file: 本地预训练权重文件路径，用于离线环境；None 则走默认下载
        """
        super().__init__()
        # num_classes=0 去掉分类头，只取 pooled 特征向量
        self.backbone = _build_backbone(backbone_name, pretrained, weights_file)
        feat_dim = self.backbone.num_features
        self.embedding = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        """
        :param x: [B, 3, 224, 224]
        :return: [B, embedding_dim] 的 L2 归一化 embedding
        """
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = F.normalize(emb, p=2, dim=1)  # L2 归一化，适配余弦相似度
        return emb


class ZeroShotBackbone(nn.Module):
    """
    零训练 baseline 用的特征提取器：预训练 backbone 的 pooled 特征 → L2 归一化。

    不含 embedding head、不参与训练，用于回答"不训练能到多少"这个对照问题。
    红线 7 要求任何训练结果都必须与这条线对比，低于它即为负优化。

    输出契约与 PetEmbeddingNet 一致（均为 L2 归一化向量），故下游评测代码
    可直接用点积当余弦相似度，无需按模式分支。
    维度随 backbone 变化（EfficientNet-B0 为 1280，Swin-T 为 768），不强制对齐到 512，
    因为一旦经过随机初始化的 Linear 降维就不再是"零训练"了。
    """

    def __init__(self, backbone_name, weights_file=None):
        """
        :param backbone_name: timm backbone 名，如 efficientnet_b0 / swin_tiny_patch4_window7_224
        :param weights_file: 本地预训练权重文件路径，用于离线环境；None 则走 timm 默认下载
        """
        super().__init__()
        self.backbone = _build_backbone(backbone_name, pretrained=True, weights_file=weights_file)

    def forward(self, x):
        """

        :param x: [B, 3, 224, 224]
        :return: [B, backbone.num_features] 的 L2 归一化特征
        """
        return F.normalize(self.backbone(x), p=2, dim=1)


def build_arcface_loss(num_classes, embedding_dim=512, margin=28.6, scale=64):
    """
    构建 ArcFace 损失（含可学习的类中心权重）。

    margin/scale 用 pytorch-metric-learning 默认推荐值起步（margin 单位为度）；
    异宠区分不够时可适当增大 margin 以拉大类间间隔。


    :param num_classes: 训练集个体数
    :return: ArcFaceLoss 实例（其参数需一并加入 optimizer）
    """
    return ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_dim,
        margin=margin,
        scale=scale,
    )


def build_triplet_loss(margin=0.3):
    """
    构建成对度量损失（Triplet + 批内难样本挖掘），替代 ArcFace 的类中心方案。

    为什么弃用类中心：ArcFace 需要 num_classes × embedding_dim 的可学习类中心，
    而公开数据集每个体仅约 3.6 张图，单个类中心拿不到足够类内信号，同时负样本
    压力随数据量线性膨胀（20,000 个体时每类正:负梯度 ≈ 1:19,258，60,000 时 1:60,272）。
    实测个体数从 20,000 涨到 40,000 即特征塌陷（余弦全挤到 1.0、AUC 倒退向 0.5），
    见技术方案附录 C.15.21。Triplet 不建类中心，直接优化「同体相似度 − 异体相似度」
    这个差值 —— 它正是评测指标 TPR@FPR 所依赖的量，故个体数不再进入优化难度。

    两处参数选择的依据：
      - distance 显式传 CosineSimilarity：PML 默认是 LpDistance，而
        PetEmbeddingNet.forward 输出已 L2 归一化、评测端用点积当余弦
        （baseline_target_domain.eval_baseline），训练与评测必须同一度量。
      - margin=0.3：NextViT 零训练实测同体余弦 0.6393 / 异体 0.1965，gap 0.4428
        （C.15.20），要求 0.3 的间隔与该量级相称，不会开局就全饱和或全失效。

    返回的 loss 无可学习参数，故【不需要】加入 optimizer（与 ArcFace 相反）。


    :param margin: 余弦相似度空间的间隔，要求 sim_pos - sim_neg >= margin
    :return: (loss_fn, miner)。miner 逐批挑最难正/负样本，调用方式为
        loss_fn(emb, labels, miner(emb, labels))
    """
    from pytorch_metric_learning.distances import CosineSimilarity
    from pytorch_metric_learning.losses import TripletMarginLoss
    from pytorch_metric_learning.miners import BatchHardMiner

    return (TripletMarginLoss(margin=margin, distance=CosineSimilarity()),
            BatchHardMiner())

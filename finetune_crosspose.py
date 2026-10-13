# -*- coding: utf-8 -*-
"""
命题方跨姿态 in-domain 微调（NextViT + Triplet，按个体划分 train/test）。

任务：宠物个体 Re-ID，交付判据「一张正脸 + 一张全身，判是不是同一只」（跨姿态判同）。
硬指标：口径 B —— TPR@FPR2% >= 95%。

为什么转到这条路（见技术方案 C.15.22）：
  此前一直用公开数据集训练、在命题方数据上评测。Triplet 换损失解决了 ArcFace 的冷启动
  损伤（e1 即 68.17%，首次超过零训练 58.40%），但两跑都是【e1 峰值、之后单调降】：
      Run A（猫狗混杂）68.17% → 67.42% → 48.54%
      Run B（同物种批）62.25% → 60.94%
  同物种采样没能阻止退化，故「批内负样本太易」不是根因。真正的病根是域不匹配：
  公开数据的图对是同姿态（脸 vs 脸），训练目标是「同姿态判同」；交付口径是跨姿态。
  预训练权重本身带一部分跨姿态不变性（零训练 58.40% 即来自此），训练越深入地拟合
  「同姿态判同」，就越多地消耗掉那部分不变性。
  而唯一 in-domain 的数据——命题方 1065 个体的真实跨姿态图——此前一张都没用于训练。

为什么用 Triplet 而非 ArcFace：
  命题方每个体【恰好 2 图】（front + overall）。K=2 的 P×K 采样下，每个 anchor 的唯一
  正样本就是另一个姿态，即正对 100% 是跨姿态对，与交付判据逐字对齐。反之 ArcFace 在
  800 类每类 2 图上，类中心信号比公开数据的 3.6 张/体更弱，正是 C.15.21 塌陷的成因。

为什么不复用 finetune_intern.py：
  它的口径是 InternViT + ArcFace + 全量无独立测集（当时用户明确要求），改它会破坏已
  记录的口径。本脚本五处全变（backbone、损失、按个体划分、起点权重、不冻结 backbone）。

⚠️ 评测口径已变，旧锚点不可直接比：
  58.40%（零训练）与 68.17%（Run A e1）都是在【全部 1065 个体】上算的；本脚本只在
  held-out 的约 264 个体上评测。故须先用 --eval_only 在同一测集上重算两个锚点
  （X=零训练、Y=Run A e1），再据此定判据。不重算就直接拿 68.17% 当门槛，
  是第四次判据未校准（前三次见 C.15.20 / C.15.21 / 用 AUC 当代理量那次）。

实测结果（2026-09-30，held-out 267 个体 / 534 对，seed=42，img_size=224）：
  锚点 X 零训练 NextViT        AUC 0.9466  TPR@FPR2% 57.30%
  锚点 Y Run A e1（公开数据）   AUC 0.9680  TPR@FPR2% 59.93%
  本脚本 20 轮（从 Y 起步）     AUC 0.978~0.983  峰值 85.77%（e16），+25.84pp
  逐轮形状：e1 81.65 → e5 85.02 → e16 85.77；e1~e10 均值 82.32%、e11~e20 均值 84.19%。
  【关键】公开数据两跑的「e1 即峰值、之后单调降」形状彻底消失，换成震荡上行后进平台，
  这是域不匹配归因的直接实验证实。同时 AUC 从 e1 起就饱和而 TPR@FPR2% 仍在涨，说明
  后续增益全部来自压制最难 2% 负对——正是决定交付指标的那部分。
  无偏估计取 e11~e20 均值 84.19%，而非 85.77%（后者用测集选了模，是乐观值）。

下一个杠杆：img_size 224 → 448。命题方原图本就是 448，224 输入把全身图里宠物脸压到
  只剩几十像素，而上面的实测指出瓶颈正在细粒度纹理。NextViT 是卷积 stem 的混合架构，
  实测吃 448 输入且 feat 维仍为 1024，故 224 训的权重可直接 --init_from。

运行方式（需 GPU；数据默认取 datasets/命题方数据集/images_448，可用 --img_root 覆盖）：
  # 第 0 步：算 held-out 测集上的两个锚点（零训练 / 起点权重）
  python finetune_crosspose.py --eval_only
  python finetune_crosspose.py --eval_only --init_from path/to/model_best.pt
  # 第 1 步：微调（224 口径，峰值 85.77%）
  python finetune_crosspose.py --init_from path/to/model_best.pt \
      --batch_size 64 --epoch_size 3200 --epochs 20 --lr 1e-3 --backbone_lr 1e-5
  # 第 2 步：分辨率分离实验——不重训，只把评测分辨率提到 448，看纯分辨率值多少 pp
  python finetune_crosspose.py --eval_only --img_size 448 \
      --init_from path/to/ckpt_crosspose/model_best.pt --eval_batch_size 16


"""
import os
import time
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import PetEmbeddingNet, build_triplet_loss
from sampler import SpeciesPKBatchSampler
from baseline_target_domain import scan_individuals, build_cross_pose_pairs, eval_baseline


def species_of(iid):
    """
    从命题方个体 id 取物种：目录名形如 0001_cat / 0003_dog，物种在下划线【后】段。

    与公开数据集的约定相反（那边是 dog_xxx，物种在前段），故 SpeciesPKBatchSampler
    必须显式注入本函数，否则会把每个 id 当成独立物种、全部剔除后抛 ValueError。

    :param iid: 个体 id（即目录名），如 "0001_cat"
    :return: 物种字符串，"cat" 或 "dog"

    """
    return iid.split("_", 1)[1]


def build_crosspose_transform(img_size, train):
    """
    构造可变分辨率的增强链，增强算子与 dataset.build_transform 逐项一致，仅 Resize 可变。

    为什么不复用 dataset.build_transform：它的 Resize 硬编码 224，而命题方原图是 448
    （pet-consistency-448-v1）。224 输入把全身图里的宠物脸压到只剩几十像素见方，
    而 C.15.22 的实测显示瓶颈正在「最难 2% 负对」——即同品种同花色个体的细粒度纹理，
    恰恰是被这次 resize 丢掉的信息。改 dataset.build_transform 会同时改掉公开数据
    那几跑的增强口径（58.40% / 68.17% 锚点皆依赖它），故在此独立定义。

    同一函数同时服务 train / eval 两态，保证两态的 Resize 尺寸不会被改漏而失配。
    CoarseDropout 的孔洞尺寸按 img_size 等比缩放（224 下 16~32px，即 7%~14% 边长），
    否则 448 下固定 16~32px 的遮挡面积只剩四分之一、遮挡增强形同虚设。

    :param img_size: 输入方形边长（224 或 448）
    :param train: 是否训练态（训练态开随机增强，评测态只 resize + 归一化）
    :return: albumentations Compose

    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if not train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # 遮挡孔洞随分辨率等比放大，保持「遮掉多少比例」不变
    lo, hi = max(int(img_size * 16 / 224), 1), max(int(img_size * 32 / 224), 2)
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.06, scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),  # 角度
        A.RandomBrightnessContrast(p=0.5),  # 光线
        A.HueSaturationValue(p=0.3),        # 光线/色彩
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(lo, hi),
                        hole_width_range=(lo, hi), p=0.3),  # 遮挡
        A.OneOf([A.MotionBlur(), A.GaussianBlur()], p=0.2),  # 拍摄模糊
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


class CrossPoseSplitDataset(Dataset):
    """
    命题方跨姿态数据集，按【个体】分层划分 train / test。

    按个体划而非按图划：同一只宠物的 front 在训练集、overall 在测试集就是泄漏——
    模型只要记住这只宠物的外观就能答对，测出来的数字不是泛化能力。

    分层（每物种各取 train_ratio）保证 train/test 的 cat:dog 比例与全集一致，
    否则测集物种失衡会让「同物种负对」的难度分布偏移、与锚点不可比。
    实测 cat 485 / dog 582，ratio=0.75 → 训练约 801（cat 364 + dog 437）/ 测试约 264。

    每个体 2 图（front + overall）各作一条样本，label = 该个体在本 split 内的连续编号。
    同时暴露 self.by_species（结构与 scan_individuals 返回一致），供评测端直接喂给
    build_cross_pose_pairs，避免为了评测再扫一遍目录。


    """

    def __init__(self, img_root, split, train_ratio=0.75, seed=42, img_size=224):
        """
        :param img_root: 命题方图片根（含 {idx}_{pet_type} 子目录）
        :param split: "train" 或 "test"
        :param train_ratio: 每物种取多少比例作训练，其余作测试
        :param seed: 划分与打乱的种子。train/test 两次构造须传【相同】seed，
            否则两边的个体集合会重叠、造成泄漏
        :param img_size: 输入方形边长，须与评测端 --img_size 及起点权重一致

        """
        if split not in ("train", "test"):
            raise ValueError(f"split 必须是 train/test，收到 {split}")

        all_by_species = scan_individuals(img_root)  # 复用：只收 front+overall 齐全的个体

        # 逐物种分层划分。先按 id 排序再用固定 seed 打乱，保证与扫盘顺序无关、可复现
        self.by_species = defaultdict(list)
        for sp, inds in sorted(all_by_species.items()):
            inds = sorted(inds, key=lambda it: it["id"])
            random.Random(seed).shuffle(inds)
            n_train = int(len(inds) * train_ratio)
            self.by_species[sp] = inds[:n_train] if split == "train" else inds[n_train:]

        # 展平后按 id 排序编连续 label（可复现，与物种遍历顺序无关）
        inds = sorted((it for lst in self.by_species.values() for it in lst),
                      key=lambda it: it["id"])
        self.id2label = {it["id"]: i for i, it in enumerate(inds)}
        self.num_classes = len(self.id2label)

        # 每个体 2 图各作一条样本。K=2 采样时同批内这两张必然同时出现，
        # 于是唯一正对就是 front↔overall，即交付判据本身
        self.samples = []
        for it in inds:
            lbl = self.id2label[it["id"]]
            self.samples.append((it["front"], lbl))
            self.samples.append((it["overall"], lbl))

        # 增强算子与公开数据那套逐项一致（含 KR5：光线/角度/遮挡/模糊），只有 Resize 随
        # img_size 变。224 时与 dataset.build_transform 完全等价，故 224 口径不漂移
        self.transform = build_crosspose_transform(img_size, train=(split == "train"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # 中文路径安全读法：数据根含「一致性识别基础数据集」，cv2.imread 在 Windows
        # 下读不了非 ASCII 路径会返回 None，故按字节读再 imdecode（与 eval_baseline 同处理）
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, label


def evaluate(model, test_set, device, seed, target_fpr, batch_size, img_size):
    """
    在 held-out 个体上按交付口径评测：正对=同体 front↔overall，负对=同物种异体。

    复用 build_cross_pose_pairs + eval_baseline，与零训练 58.40% 那条线同源，
    差别仅在个体范围（此处只用 held-out 那约 264 个，不含任何训练个体）。

    ⚠️ 测集粒度粗：约 264 负对下 FPR2% 只落在最难的约 5.3 个负对上，TPR 粒度约 0.38%，
    逐轮曲线会跳。判读看趋势与峰值，单轮 ±1pp 不作为信号。

    eval_baseline 自带 @torch.no_grad 且内部置 eval，但【不恢复 train 模式】，
    故调用方在训练循环里评完须手动 model.train()。

    :param model: 输出 L2 归一化 embedding 的模型（可能被 DataParallel 包裹）
    :param test_set: CrossPoseSplitDataset(split="test")，取其 by_species
    :param device: 推理设备
    :param seed: 负对采样种子，固定以保证逐轮可比
    :param target_fpr: 目标假正率，0.02 对应异宠区分率 98%
    :param batch_size: 提特征批大小
    :param img_size: 输入边长，须与训练/起点权重一致
    :return: (auc, tpr_at_fpr, threshold, n_pairs)

    """
    pairs, labels = build_cross_pose_pairs(test_set.by_species, seed)
    auc, tpr, thr = eval_baseline(model, pairs, labels, device,
                                  target_fpr=target_fpr, batch_size=batch_size,
                                  img_size=img_size)
    return auc, tpr, thr, len(pairs)


def parse_args():
    """"""
    # 默认数据/输出路径都相对脚本自身位置解析，使团队成员无论从仓库根还是 code/ 下
    # 运行都能定位到 datasets/，避免硬编码绝对路径导致换机即跑不通
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--img_root",
                   default=os.path.join(here, "datasets", "命题方数据集", "images_448"),
                   help="命题方图片根（含 {idx}_{pet_type} 子目录）")
    p.add_argument("--out_dir", default=os.path.join(here, "ckpt_crosspose"))
    p.add_argument("--weights", default=None,
                   help="backbone 本地预训练权重（离线用）；联网时留空走 timm 自动下载")
    p.add_argument("--init_from", default=None,
                   help="起点权重 ckpt（PetEmbeddingNet 完整 state_dict，存于 ck['model']）。"
                        "留空则从 ImageNet 预训练权重起步")

    p.add_argument("--backbone", default="nextvit_base.bd_ssld_6m_in1k")
    p.add_argument("--img_size", type=int, default=224,
                   help="输入方形边长，训练与评测同时生效（见 build_crosspose_transform）。"
                        "224 与 Run A e1 起点权重一致；命题方原图是 448，提到 448 可把"
                        "全身图里宠物脸的有效像素提高 4 倍（C.15.22 平台期后的下一个杠杆）。"
                        "⚠️ 改了它就与 224 下的锚点不同口径，须重算锚点方可比")
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--train_ratio", type=float, default=0.75,
                   help="每物种取多少比例作训练（0.75 → 约 801 训 / 264 测）")

    p.add_argument("--epochs", type=int, default=20, help="总轮数，同时是 cosine 的 T_max")
    p.add_argument("--batch_size", type=int, default=64,
                   help="即 P×K；64 → P=32 个体。批太小则 BatchHardMiner 挑不到难负样本")
    p.add_argument("--epoch_size", type=int, default=3200,
                   help="每轮采样图片数，决定每轮步数（3200/64 = 50 步）。"
                        "训练集仅约 1602 图，取 2 倍即每图每轮约采 2 次")
    p.add_argument("--k_per_id", type=int, default=2,
                   help="每个体每批取几张。命题方每体恰好 2 图，K=2 时正对 100% 跨姿态；"
                        "K>2 会触发有放回采样、产生同图正对（损失恒为 0，白占算力）")
    p.add_argument("--margin", type=float, default=0.3, help="triplet 余弦间隔")
    p.add_argument("--lr", type=float, default=1e-3, help="head(embedding) 学习率")
    p.add_argument("--backbone_lr", type=float, default=1e-5,
                   help="backbone 学习率。比公开数据那跑（1e-4）再小 10 倍——"
                        "训练集仅约 1602 图 / backbone 43.8M 参数，过拟合是主要风险")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_fpr", type=float, default=0.02)
    p.add_argument("--eval_batch_size", type=int, default=64, help="评测提特征批大小")
    p.add_argument("--eval_only", action="store_true",
                   help="只在 held-out 测集上评测一次就退出，用于算锚点（第 0 步）")
    return p.parse_args()


def main():
    """
    主流程：按个体分层划分 → 建模型（可从公开数据权重起步）→ 同物种 P×K + Triplet 微调
    → 每轮在 held-out 个体上按交付口径评测 → 存 ckpt。

    Triplet 损失无可学习参数，故 optimizer 只有 backbone / embedding 两组，
    不能像 ArcFace 那样追加 criterion 参数组。


    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # 1. 划分：train/test 传同一 seed，保证个体集合互补且无交集
    t0 = time.time()
    train_set = CrossPoseSplitDataset(args.img_root, "train", args.train_ratio,
                                      args.seed, args.img_size)
    test_set = CrossPoseSplitDataset(args.img_root, "test", args.train_ratio,
                                     args.seed, args.img_size)
    overlap = set(train_set.id2label) & set(test_set.id2label)
    if overlap:  # fail loud：泄漏会让后续所有数字失去意义
        raise RuntimeError(f"train/test 个体交集非空（{len(overlap)} 个），划分有 bug")
    print(f"[INFO] 扫数据+划分 {time.time() - t0:.1f}s")
    print(f"[INFO] 训练 {train_set.num_classes} 个体 / {len(train_set)} 图  "
          + "  ".join(f"{k}={len(v)}" for k, v in sorted(train_set.by_species.items())))
    print(f"[INFO] 测试 {test_set.num_classes} 个体 / {len(test_set)} 图  "
          + "  ".join(f"{k}={len(v)}" for k, v in sorted(test_set.by_species.items())))
    print("[INFO] train/test 个体交集为 0，测集零泄漏")

    # 2. 模型：NextViT + embedding head，可从公开数据训练过的权重起步
    print(f"[INFO] backbone 预训练权重: {args.weights or 'timm 在线下载'}")
    model = PetEmbeddingNet(args.backbone, args.embedding_dim, pretrained=True,
                            weights_file=args.weights).to(device)
    if args.init_from:
        # weights_only=False：ckpt 存了 val_tpr 等 numpy 标量，PyTorch 2.6 起默认
        # weights_only=True 会拒绝反序列化；该 ckpt 为本项目自存、来源可信
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[INFO] 起点权重 {args.init_from}"
              f"（backbone={ck.get('backbone')} epoch={ck.get('epoch')} "
              f"tpr={ck.get('val_tpr')}）")
    else:
        print("[INFO] 起点权重：ImageNet 预训练（未经公开数据训练）")

    # 3. eval_only：只算锚点。放在建 optimizer/DataParallel 之前，避免无谓开销
    if args.eval_only:
        auc, tpr, thr, n_pairs = evaluate(model, test_set, device, args.seed,
                                          args.target_fpr, args.eval_batch_size,
                                          args.img_size)
        print(f"\n[锚点] held-out {test_set.num_classes} 个体 / {n_pairs} 对（跨姿态）")
        print(f"  AUC={auc:.4f}  TPR@FPR{args.target_fpr:.0%}={tpr:.4f}  阈值={thr:.4f}")
        print("[提示] 该数与全量 1065 个体上的 58.40% / 68.17% 口径不同，不可直接比；"
              "本数即新测集上的对照锚点")
        return

    # 4. 同物种 P×K 采样：批内只在单一物种内组批，使 miner 挑到的最难负样本是同物种异体，
    #    与评测负对口径（100% 同物种）对齐。命题方 id 是 0001_cat，物种在后段，故传 species_of
    batch_sampler = SpeciesPKBatchSampler(train_set, args.k_per_id, args.batch_size,
                                          args.epoch_size, args.seed,
                                          species_fn=species_of)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    criterion, miner = build_triplet_loss(args.margin)
    criterion = criterion.to(device)
    print(f"[INFO] loss=triplet(margin={args.margin}, K={args.k_per_id}, "
          f"P={args.batch_size // args.k_per_id}, 批内负样本=同物种)"
          f"  每轮 {len(train_loader)} 步")

    # 5. 分组学习率（红线 4）：backbone 小 lr、embedding 大 lr。
    #    Triplet 无可学习参数，故只有 2 组（ArcFace 那边是 3 组）。
    #    优化器分组须在包 DataParallel 之前建——包了之后子模块要走 model.module.xxx
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.embedding.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    print(f"[INFO] lr: backbone={args.backbone_lr}  head={args.lr}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        print(f"[INFO] DataParallel 启用，GPU 数={n_gpu}")

    print("[提醒] 训练集仅约 1602 图，过拟合会很快；每轮评测 + 存 best，峰值轮次本身即结论")

    # 6. 训练循环
    best_tpr = -1.0
    for epoch in range(args.epochs):
        model.train()
        running, t_ep = 0.0, time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            emb = model(imgs)
            loss = criterion(emb, labels, miner(emb, labels))  # 先挑最难三元组再算损失
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        avg = running / max(len(train_loader), 1)

        auc, tpr, thr, _ = evaluate(model, test_set, device, args.seed,
                                    args.target_fpr, args.eval_batch_size, args.img_size)
        model.train()  # eval_baseline 不恢复 train 模式，手动恢复
        print(f"[epoch {epoch + 1}] loss={avg:.4f}  test_AUC={auc:.4f}  "
              f"test_TPR@FPR2%={tpr:.4f}  阈值={thr:.4f}  ({time.time() - t_ep:.0f}s)")

        # 存 ckpt。DP 下取 model.module（去掉 module. 前缀），与 --init_from 的读法对齐。
        # 不存 optimizer/scheduler：20 轮训练耗时短，一次跑完即可，无需分段续训
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        ckpt = {"model": core.state_dict(),
                "backbone": args.backbone, "embedding_dim": args.embedding_dim,
                "img_size": args.img_size,
                "epoch": epoch + 1, "val_tpr": tpr, "best_tpr": max(best_tpr, tpr)}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if tpr > best_tpr:
            best_tpr = tpr
            torch.save(ckpt, os.path.join(args.out_dir, "model_best.pt"))

    print(f"[INFO] 跑完 {args.epochs} 轮，best test_TPR@FPR2%={best_tpr:.4f}"
          f"（held-out {test_set.num_classes} 个体，零泄漏）；ckpt 在 {args.out_dir}")


if __name__ == "__main__":
    main()

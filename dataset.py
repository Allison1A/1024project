"""
宠物个体识别 - 数据集模块（阶段 1：单分支正脸 baseline）

核心职责：
1. 扫描个体级标注目录（每个个体一个文件夹，内含该个体多张照片）
2. 按【个体】划分 train/val/test —— 训练/测试个体绝不重叠（防数据泄漏，这是红线 1）
3. 提供训练用 Dataset（返回 image, label）和评测用 pair 构造能力

目录约定（个体级标注）：
    data_root/
        individual_0001/  img1.jpg img2.jpg ...
        individual_0002/  img1.jpg ...
        ...


"""
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# ImageNet 归一化参数（EfficientNet 预训练权重基于此）
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# 支持的图片扩展名
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def scan_individuals(data_root):
    """
    扫描个体级目录，返回 {individual_id: [图片绝对路径, ...]}。

    只保留至少含 2 张图的个体（个体识别至少需要一张做 anchor、一张做正样本）。


    :param data_root: 数据根目录，每个子文件夹是一个个体
    :return: dict[str, list[str]]
    """
    id2paths = defaultdict(list)
    for name in sorted(os.listdir(data_root)):
        sub = os.path.join(data_root, name)
        if not os.path.isdir(sub):
            continue
        for f in sorted(os.listdir(sub)):
            if f.lower().endswith(_IMG_EXTS):
                id2paths[name].append(os.path.join(sub, f))
    # 过滤掉只有 1 张图的个体
    id2paths = {k: v for k, v in id2paths.items() if len(v) >= 2}
    return id2paths


def split_by_individual(id2paths, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按【个体】划分数据集 —— 同一个体的所有图片只会出现在同一个 split。

    这是防止数据泄漏的关键：若按图片划分，同一只宠物会同时出现在训练和测试集，
    导致测试指标虚高，现场换新宠物直接崩（红线 1）。


    :return: (train_ids, val_ids, test_ids) 三个个体 id 列表
    """
    ids = sorted(id2paths.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    return train_ids, val_ids, test_ids


def build_transform(train):
    """
    构建图像增强。训练态包含对应 KR5（光线/角度/遮挡）的增强算子。


    :param train: 是否训练态（训练态开启随机增强，验证/测试态只 resize+归一化）
    """
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.06, scale=(0.9, 1.1), rotate=(-20, 20), p=0.5),  # 角度
            A.RandomBrightnessContrast(p=0.5),  # 光线
            A.HueSaturationValue(p=0.3),        # 光线/色彩
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32),
                            hole_width_range=(16, 32), p=0.3),  # 遮挡
            A.OneOf([A.MotionBlur(), A.GaussianBlur()], p=0.2),  # 拍摄模糊
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


class PetIndividualDataset(Dataset):
    """
    单分支个体识别数据集。返回 (image_tensor, label_int)。

    label 是【连续整数编码】的个体 id，供 ArcFace 分类头使用；
    只在传入的个体范围（train_ids）内构建 label 空间。


    """

    def __init__(self, id2paths, use_ids, train):
        """

        :param id2paths: scan_individuals 的返回
        :param use_ids: 本 split 使用的个体 id 列表
        :param train: 是否训练态（决定增强）
        """
        self.samples = []  # [(img_path, label_int), ...]
        self.id2label = {iid: i for i, iid in enumerate(sorted(use_ids))}
        self.num_classes = len(self.id2label)
        for iid in use_ids:
            label = self.id2label[iid]
            for p in id2paths[iid]:
                self.samples.append((p, label))
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, label


def build_eval_pairs(id2paths, use_ids, num_pos=3000, num_neg=3000, seed=42):
    """
    为评测构造同宠/异宠图片对。

    - 正样本对（同宠）：同一个体的两张不同图片 → 期望相似度高
    - 负样本对（异宠）：不同个体的两张图片 → 期望相似度低
      （注：阶段 1 随机采负样本；阶段 2 需刻意采【同品种不同个体】的困难负样本）


    :return: (pairs, labels)，pairs=[(pathA, pathB), ...]，labels=[1同宠/0异宠, ...]
    """
    rng = random.Random(seed)
    ids = [i for i in use_ids if len(id2paths[i]) >= 2]

    pairs, labels = [], []
    # 正样本对
    for _ in range(num_pos):
        iid = rng.choice(ids)
        a, b = rng.sample(id2paths[iid], 2)
        pairs.append((a, b))
        labels.append(1)
    # 负样本对
    for _ in range(num_neg):
        i1, i2 = rng.sample(ids, 2)
        a = rng.choice(id2paths[i1])
        b = rng.choice(id2paths[i2])
        pairs.append((a, b))
        labels.append(0)

    return pairs, labels

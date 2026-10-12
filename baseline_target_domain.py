# -*- coding: utf-8 -*-
"""
目标域零训练 baseline —— 命题方真实数据（正脸↔全身跨姿态判同）。

为什么单独测这条：
  第 1 步的 baseline（TPR@FPR2%=48.90%）是在公开数据集上、且是"同类型照片配对"口径；
  而命题方交付判据是「一张正脸 + 一张全身，判是不是同一只」——跨姿态配对，口径完全不同。
  跨姿态（脸部特征 vs 体型/毛色/姿态）视觉重叠小，难度更高，公开数据的 48.90% 不适用于目标域。
  本脚本用命题方 1067 个体真实数据，测现有零训练 EfficientNet-B0 特征在目标域的真实起点，
  作为后续国产 backbone + 微调的对比锚点（红线 7：任何训练结果须与零训练 baseline 比）。

评测对构造（对齐 split_pairs.json 负样本口径 + 交付判据）：
  - 正对：同个体的 front ↔ overall（跨姿态，与交付判据一致）
  - 负对：不同个体、【同物种】的 front ↔ overall（split 负样本 100% 同物种）
  - 正负等量，随机种子固定，可复现


"""
import os
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch

from model import ZeroShotBackbone
from dataset import build_transform

# 命题方图片根，相对脚本自身位置解析，使团队成员换机/换工作目录都能定位
IMG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "datasets", "命题方数据集", "images_448")


def scan_individuals(img_root):
    """
    扫描图片根目录，收集每个个体的 front/overall 路径，按物种分组。

    目录名格式为 {idx:04d}_{pet_type}（如 0001_cat / 0003_dog），
    物种取目录名下划线后段；只收 front 和 overall 都存在且非空的个体，
    缺任一张的个体无法构成跨姿态对，跳过。

    :param img_root: 图片根目录
    :return: dict[str, list[dict]]，species -> [{id, front, overall}, ...]

    """
    by_species = defaultdict(list)
    for name in sorted(os.listdir(img_root)):
        d = os.path.join(img_root, name)
        if not os.path.isdir(d):
            continue
        species = name.split("_", 1)[1] if "_" in name else "unknown"
        front = os.path.join(d, "front.jpg")
        overall = os.path.join(d, "overall.jpg")
        if (os.path.isfile(front) and os.path.getsize(front) > 0
                and os.path.isfile(overall) and os.path.getsize(overall) > 0):
            by_species[species].append({"id": name, "front": front, "overall": overall})
    return by_species


def build_cross_pose_pairs(by_species, seed):
    """
    构造跨姿态评测对：正对 = 同体 front↔overall；负对 = 同物种异体 front↔overall。

    正对数 = 有效个体数（每个体一对）；负对数取等量，从同物种内随机配对不同个体
    （A 的 front vs B 的 overall），保证与 split 口径一致的"负样本同物种"。

    :param by_species: scan_individuals 的返回
    :param seed: 随机种子，固定负对采样，保证可复现
    :return: (pairs, labels)，pairs=[(pathA, pathB), ...]，labels=[1/0, ...]

    """
    rng = random.Random(seed)
    pos, neg = [], []
    for species, inds in by_species.items():
        # 正对：每个体 front↔overall
        for it in inds:
            pos.append((it["front"], it["overall"]))
        # 负对：同物种内，A.front vs B.overall（A≠B），采与该物种正对等量
        n = len(inds)
        if n < 2:
            continue
        cnt = 0
        while cnt < n:
            a, b = rng.sample(range(n), 2)
            neg.append((inds[a]["front"], inds[b]["overall"]))
            cnt += 1
    pairs = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return pairs, labels


def build_eval_transform(img_size):
    """
    构造评测态图像变换。img_size==224 时复用 dataset.build_transform（保持与 B0 口径完全一致）；
    否则（如 InternViT-300M 要 448 输入）自建同 ImageNet 归一化的 resize 变换，
    只改分辨率、不改归一化，避免动 dataset.py 破坏现有 B0 训练/评测口径。

    :param img_size: 目标方形边长（B0=224，InternViT-300M=448）
    :return: albumentations Compose

    """
    if img_size == 224:
        return build_transform(train=False)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def eval_baseline(model, pairs, labels, device, target_fpr=0.02, batch_size=128, img_size=224):
    """
    在跨姿态评测对上算 AUC 与 FPR=target 处 TPR，与第 1 步 baseline 口径一致。

    同图可能被多对引用，先按唯一路径解码提特征；解码失败的图跳过并剔除受影响对。
    embedding 已 L2 归一化，点积即余弦。

    :param model: 输出 L2 归一化 embedding 的模型（ZeroShotBackbone）
    :param pairs: [(pathA, pathB), ...]
    :param labels: [1/0, ...]
    :param device: 推理设备
    :param target_fpr: 目标假正率，0.02 对应异宠区分率 98%
    :param batch_size: 提特征批大小
    :return: (auc, tpr_at_fpr, threshold)

    """
    from sklearn.metrics import roc_auc_score, roc_curve

    transform = build_eval_transform(img_size)
    uniq = sorted({p for ab in pairs for p in ab})
    rgb_by_path, valid, bad = {}, [], set()
    for p in uniq:
        # Windows 下 cv2.imread 读不了中文路径（本数据根含"一致性识别基础数据集"），
        # 改用 np.fromfile 按字节读再 imdecode，绕过 OpenCV 的路径编码限制。
        im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            bad.add(p)
            continue
        rgb_by_path[p] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        valid.append(p)
    path2idx = {p: i for i, p in enumerate(valid)}

    kept = [(a, b, y) for (a, b), y in zip(pairs, labels) if a not in bad and b not in bad]
    dropped = len(pairs) - len(kept)
    if dropped:
        print(f"[eval] 坏图 {len(bad)} 张，剔除受影响对 {dropped}/{len(pairs)}")
    pairs = [(a, b) for a, b, _ in kept]
    labels = [y for _, _, y in kept]

    model.eval()
    embs = []
    n_batches = (len(valid) + batch_size - 1) // batch_size
    for bi, i in enumerate(range(0, len(valid), batch_size), 1):
        batch = torch.stack([
            transform(image=rgb_by_path[p])["image"]
            for p in valid[i:i + batch_size]
        ]).to(device)
        embs.append(model(batch).cpu().numpy())
        print(f"  提特征 batch {bi}/{n_batches}", flush=True)  # CPU 慢，实时进度防误判卡死
    embs = np.concatenate(embs)

    scores = np.array([float(embs[path2idx[a]] @ embs[path2idx[b]]) for a, b in pairs])
    y = np.array(labels)
    auc_val = roc_auc_score(y, scores)
    fpr, tpr, thr = roc_curve(y, scores)
    idx = max(int(np.searchsorted(fpr, target_fpr, side="right") - 1), 0)
    return auc_val, tpr[idx], thr[idx]


def main():
    """
    主流程：扫个体 → 构跨姿态正负对 → 零训练 B0 提特征评测 → 打印目标域起点。


    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", default=IMG_ROOT)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--weights", default=None, help="本地预训练权重；None 走 timm 下载")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_fpr", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="CPU 提特征批大小，本机内存小时调小防卡死")
    ap.add_argument("--max_ind", type=int, default=None,
                    help="每物种只取前 N 个体，用于先小规模验证不卡死；None 为全量")
    ap.add_argument("--img_size", type=int, default=224,
                    help="输入分辨率。B0=224；InternViT-300M(vit_intern300m_patch14_448)须传 448")
    args = ap.parse_args()

    # CPU 版 torch 默认吃满全部核心，B0 逐批前向易把本机 CPU/内存打满卡死；
    # 限到 4 线程，牺牲一点速度换机器不卡死（本任务仅 2130 张，慢一点可接受）。
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  torch_threads={torch.get_num_threads()}")

    by_species = scan_individuals(args.img_root)
    if args.max_ind is not None:  # 小规模验证：每物种只留前 N 个体
        by_species = {k: v[:args.max_ind] for k, v in by_species.items()}
    total = sum(len(v) for v in by_species.values())
    print(f"[INFO] 有效个体（front+overall 齐全）={total}  "
          + "  ".join(f"{k}={len(v)}" for k, v in by_species.items()))

    pairs, labels = build_cross_pose_pairs(by_species, args.seed)
    print(f"[INFO] 评测对：正 {sum(labels)}  负 {len(labels) - sum(labels)}  共 {len(labels)}")

    model = ZeroShotBackbone(args.backbone, weights_file=args.weights).to(device)
    auc, tpr, thr = eval_baseline(model, pairs, labels, device,
                                  args.target_fpr, args.batch_size, args.img_size)

    print(f"\n[目标域 baseline] 零训练 {args.backbone}（跨姿态 正脸-全身）")
    print(f"  AUC={auc:.4f}  TPR@FPR{args.target_fpr:.0%}={tpr:.4f}  阈值={thr:.4f}")
    print(f"[对照] 公开数据同类型配对 baseline TPR@FPR2%=48.90% / AUC=0.9091（口径不同，仅参考）")


if __name__ == "__main__":
    main()

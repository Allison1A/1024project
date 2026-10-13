"""
按命题承诺口径（异宠区分率 ≥ 98%，即 FAR ≤ 2%）重算各场景的同宠识别率。

动机：此前所有实验都报在 FPR=1%（stage0.roc_metrics 只评 1e-2/1e-3 两档），
而命题验收标准写的是「同宠识别率 ≥95% 且异宠区分率 ≥98%」——后者对应 FAR=2%。
口径不对齐则初审稿里的数字无法与验收标准直接对照。本脚本不改动 stage0，
在同一套正负样本上额外取 FAR=2% 工作点。

零推理成本：全部复用已落盘的嵌入。

@author kk
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from stage0_detect_crop import (
    CACHE_DIR,
    EMB_CROP_PATH,
    EMB_FULL_PATH,
    MANIFEST_PATH,
    build_scores,
    l2norm,
)
from stage2_asym_combo import load_full_baseline
from stage3_dogfacenet_baseline import (
    EMB_PATH as DFN_EMB_PATH,
    IDS_PATH as DFN_IDS_PATH,
    MAX_NEG_PAIRS,
    build_neg_pairs,
    build_pos_pairs,
)

# 命题验收标准：异宠区分率 ≥98% / ≥99% 分别对应 FAR=2% / 1%
FAR_TARGETS = (0.02, 0.01, 0.001)


def tpr_at_far(pos: np.ndarray, neg: np.ndarray, far: float) -> Tuple[float, float]:
    """
    取指定 FAR 下的阈值与对应 TPR。

    阈值按负样本的 (1-far) 分位取，与 stage0.roc_metrics 的实现口径一致，
    保证 FAR=1% 的结果能与此前所有数字对上。

    @author kk
    """
    thr = float(np.percentile(neg, (1.0 - far) * 100.0))
    return float((pos >= thr).mean()), thr


def report(tag: str, pos: np.ndarray, neg: np.ndarray) -> Dict[float, Tuple[float, float]]:
    """
    打印单一场景在各 FAR 档位下的同宠识别率。

    @author kk
    """
    print(f"\n--- {tag} ---")
    print(f"正样本 {len(pos):>8} 对   均值 {pos.mean():.4f}")
    print(f"负样本 {len(neg):>8} 对   均值 {neg.mean():.4f}")
    out: Dict[float, Tuple[float, float]] = {}
    for far in FAR_TARGETS:
        tpr, thr = tpr_at_far(pos, neg, far)
        out[far] = (tpr, thr)
        print(f"  异宠区分 {(1 - far) * 100:>5.1f}%（FAR={far:.3%}）"
              f" → 同宠识别 {tpr * 100:>6.2f}%   阈值 {thr:.4f}")
    return out


def load_business_embedding(emb_path: str, manifest: pd.DataFrame) -> np.ndarray:
    """
    载入业务数据嵌入并对齐到清洗后清单的顺序。

    整图嵌入按下载阶段的旧清单顺序存储（2130 行，含后来被剔除的 63 行自重复等），
    需按行号重建索引；裁剪类嵌入已是清洗后顺序，直接用。

    @author kk
    """
    if emb_path == EMB_FULL_PATH:
        return load_full_baseline(manifest)
    emb = np.load(emb_path)
    if emb.shape[0] != 2 * len(manifest):
        raise ValueError(f"嵌入 {emb.shape[0]} 行与清单 {2 * len(manifest)} 张不匹配")
    return emb


def business_scores(emb_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    载入业务数据嵌入并按 stage0 口径构造正负样本，附带 pet_type 供分物种拆分。

    @author kk
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    pet_types = manifest["pet_type"].to_numpy()
    emb = load_business_embedding(emb_path, manifest)
    pos, neg = build_scores(l2norm(emb), pet_types)
    return pos, neg, pet_types


def species_split(emb_path: str, species: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单一物种子集内构造正负样本，口径与 stage4 一致。

    @author kk
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    mask = (manifest["pet_type"] == species).to_numpy()
    emb = l2norm(load_business_embedding(emb_path, manifest))
    front, overall = emb[0::2][mask], emb[1::2][mask]
    pos = np.einsum("ij,ij->i", front, overall)
    sim = np.maximum(front @ front.T, overall @ overall.T)
    iu = np.triu_indices(sim.shape[0], k=1)
    return pos, sim[iu]


def main() -> None:
    """
    入口：业务数据（整图/全身框/分物种）与 DogFaceNet 全部按 FAR=2% 重报。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print("===== 按命题承诺口径重报（异宠区分 ≥98% ⇔ FAR ≤ 2%）=====")

    rows: List[Tuple[str, Dict[float, Tuple[float, float]]]] = []

    pos, neg, _ = business_scores(EMB_FULL_PATH)
    rows.append(("业务 整图", report("业务数据 正脸↔全身（整图）", pos, neg)))

    pos, neg, _ = business_scores(EMB_CROP_PATH)
    rows.append(("业务 全身框", report("业务数据 正脸↔全身（全身框裁剪）", pos, neg)))

    for sp, label in (("cat", "猫"), ("dog", "狗")):
        p, n = species_split(EMB_CROP_PATH, sp)
        rows.append((f"业务 全身框·{label}", report(f"业务数据 {label}（全身框裁剪）", p, n)))

    if os.path.exists(DFN_EMB_PATH) and os.path.exists(DFN_IDS_PATH):
        emb = l2norm(np.load(DFN_EMB_PATH))
        ids = np.load(DFN_IDS_PATH)
        p = build_pos_pairs(emb, ids)
        n = build_neg_pairs(emb, ids, MAX_NEG_PAIRS)
        rows.append(("DogFaceNet 脸↔脸", report("DogFaceNet 跨场景（脸↔脸）", p, n)))
    else:
        print("\n[跳过] DogFaceNet 嵌入缓存不存在，先跑 stage3_dogfacenet_baseline.py")

    print("\n===== 汇总：同宠识别率（TAR）=====")
    print(f"{'场景':<22}{'FAR=2%':>10}{'FAR=1%':>10}{'FAR=0.1%':>10}")
    for tag, m in rows:
        print(f"{tag:<22}{m[0.02][0] * 100:>9.2f}%{m[0.01][0] * 100:>9.2f}%"
              f"{m[0.001][0] * 100:>9.2f}%")
    print("\n承诺口径：同宠识别 ≥95% @ 异宠区分 ≥98%（即 FAR=2% 列须 ≥95%）")


if __name__ == "__main__":
    main()

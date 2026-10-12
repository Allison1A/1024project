"""
Stage 2 —— 非对称裁剪组合实验。

动机：Stage 1 的裁头预览暴露了一个不对称事实——**正脸照里头部本来就占满画面**，
再取全身框上部反而会切进脸里（实测 row3 只裁到耳朵）；真正需要裁头的是全身照。
所以「两侧都裁头」未必最优，「front 用全身框 / overall 裁头」可能更好。

本脚本穷举 front 与 overall 各自的裁剪方式组合，评测哪种配对最优。零额外推理成本：
所有嵌入都已由 Stage 0/1 落盘，这里只是重新配对已有向量。

评测口径与 Stage 0/1 一致：
  - 正样本 = 行内 front(方式 A) 与 overall(方式 B) 的相似度
  - 负样本 = 跨行同姿态 max(front_A 互比, overall_B 互比)，限同 pet_type
    注意负样本必须用「各自方式内部互比」，跨方式比会把裁剪差异混入身份判断

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
    l2norm,
    roc_metrics,
)
from stage1_head_crop import TOP_RATIOS, ratio_tag

# 候选裁剪方式 → 嵌入缓存文件名
VARIANTS: Dict[str, str] = {
    "整图": EMB_FULL_PATH,
    "全身框": EMB_CROP_PATH,
    **{f"头{r}": os.path.join(CACHE_DIR, f"small_cls_{ratio_tag(r)}.npy") for r in TOP_RATIOS},
}


def load_full_baseline(manifest: pd.DataFrame) -> np.ndarray:
    """
    读取整图基线嵌入并对齐到当前清单顺序。

    整图嵌入按下载阶段的旧清单顺序存储（含后来被剔除的行），需按行号重建索引。

    @author kk
    """
    old = pd.read_csv(os.path.join(CACHE_DIR, "manifest.csv"))
    old = old[old.front_ok & old.overall_ok].reset_index(drop=True)
    pos_in_old = {int(r): i for i, r in enumerate(old["row"])}
    raw = np.load(EMB_FULL_PATH)

    idx: List[int] = []
    for r in manifest["row"]:
        i = pos_in_old[int(r)]
        idx.extend([2 * i, 2 * i + 1])
    return raw[idx]


def load_variants(manifest: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    载入所有可用裁剪方式的嵌入，拆成 (front, overall) 并 L2 归一化。

    缺失的缓存文件跳过而非报错，便于在 Stage 1 部分 ratio 尚未跑完时先看结果。

    @author kk
    """
    n = len(manifest)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, path in VARIANTS.items():
        if not os.path.exists(path):
            print(f"      跳过 {name}：缓存缺失 {os.path.basename(path)}")
            continue
        emb = load_full_baseline(manifest) if name == "整图" else np.load(path)
        if emb.shape[0] != 2 * n:
            print(f"      跳过 {name}：嵌入 {emb.shape[0]} 行与清单 {2 * n} 不匹配")
            continue
        emb = l2norm(emb)
        out[name] = (emb[0::2], emb[1::2])
    return out


def combo_metrics(
    front: np.ndarray, overall: np.ndarray, pet_types: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    评测一组 front/overall 裁剪方式组合，返回指标与正样本相似度数组。

    @author kk
    """
    pos = np.einsum("ij,ij->i", front, overall)

    sim = np.maximum(front @ front.T, overall @ overall.T)
    iu = np.triu_indices(sim.shape[0], k=1)
    same_type = pet_types[iu[0]] == pet_types[iu[1]]
    neg = sim[iu][same_type]
    return roc_metrics(pos, neg), pos


def main() -> None:
    """
    入口：载入各裁剪方式嵌入 → 穷举 front×overall 组合 → 按 TPR@FPR=1% 排序输出。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    manifest = pd.read_csv(MANIFEST_PATH)
    pet_types = manifest["pet_type"].to_numpy()
    print(f"清单记录 {len(manifest)} 行")

    print("\n[1/2] 载入嵌入")
    variants = load_variants(manifest)
    print(f"      可用裁剪方式 {len(variants)} 种：{'、'.join(variants)}")

    print("\n[2/2] 穷举组合")
    rows: List[Tuple[str, str, Dict[str, float], np.ndarray]] = []
    for fname, (front, _) in variants.items():
        for oname, (_, overall) in variants.items():
            m, pos = combo_metrics(front, overall, pet_types)
            rows.append((fname, oname, m, pos))

    rows.sort(key=lambda r: r[2]["tpr@fpr0.01"], reverse=True)

    print(f"\n{'front':<10}{'overall':<10}{'AUC':>8}{'EER':>8}"
          f"{'TPR@1%':>9}{'TPR@0.1%':>10}{'正样本均值':>11}")
    for fname, oname, m, pos in rows:
        print(f"{fname:<10}{oname:<10}{m['auc']:>8.4f}{m['eer'] * 100:>7.2f}%"
              f"{m['tpr@fpr0.01'] * 100:>8.2f}%{m['tpr@fpr0.001'] * 100:>9.2f}%"
              f"{pos.mean():>11.4f}")

    bf, bo, bm, bpos = rows[0]
    print(f"\n最佳组合：front={bf} / overall={bo}")
    print(f"  TPR@FPR=1% {bm['tpr@fpr0.01'] * 100:.2f}%"
          f"（阈值 {bm['thr@fpr0.01']:.4f}）  正样本均值 {bpos.mean():.4f}")
    for t in ("cat", "dog"):
        sub = bpos[pet_types == t]
        print(f"  {t:<4} 正样本均值 {sub.mean():.4f}  P10 {np.percentile(sub, 10):.4f}")

    sym = [r for r in rows if r[0] == r[1]]
    if sym:
        sf, so, sm, _ = max(sym, key=lambda r: r[2]["tpr@fpr0.01"])
        gain = bm["tpr@fpr0.01"] - sm["tpr@fpr0.01"]
        print(f"\n最佳对称组合：{sf}（TPR@1% {sm['tpr@fpr0.01'] * 100:.2f}%）"
              f"，非对称增益 {gain * 100:+.2f}pp")


if __name__ == "__main__":
    main()

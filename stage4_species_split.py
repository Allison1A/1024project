"""
Stage 4 —— 猫狗分物种指标拆分。

动机：此前所有评测都把猫狗混在一起报（TPR@FPR=1% = 50.8%），只在正样本均值上分过档
（cat 0.6954 / dog 0.7159）。但均值差异说明不了阈值层面的可用性——真正要回答的是
「猫的瓶颈和狗是不是同一个」。

这关系到训练数据的决策：stage3 证明狗的瓶颈是跨姿态（脸↔脸 85% vs 正脸↔全身 50.8%），
而 DogFaceNet 只有狗。若猫的失效模式与狗一致，则业务数据里 355 个猫 iid 的「正脸+全身」
配对就是对口的监督信号，猫并非无解；若猫另有失效模式（例如负样本压不下去——花色相近的
狸花猫互相撞脸），那就是另一个问题，得单独想办法。

拆分口径说明：负样本本来就限同 pet_type（见 stage0.build_scores），所以按物种切分后
负样本集合与混合评测完全一致，只是不再把猫狗两个子集并起来算 ROC。正样本则是各自子集。
因此本脚本零推理成本，只是对现成嵌入重新分组统计。

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
    MANIFEST_PATH,
    l2norm,
    roc_metrics,
)

# stage3 实测的狗脸↔脸跨场景水平，作为「姿态被控制住之后能到哪」的锚点
DOGFACENET_TPR_AT_1PCT = 0.8511


def load_business_embeddings(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    载入业务数据的全身框裁剪嵌入并拆成 (front, overall)，均已 L2 归一化。

    选全身框而非整图或裁头：stage1/stage2 已证明全身框是当前最优裁剪方式。

    @author kk
    """
    emb = np.load(EMB_CROP_PATH)
    if emb.shape[0] != 2 * len(manifest):
        raise ValueError(f"嵌入 {emb.shape[0]} 行与清单 {2 * len(manifest)} 张不匹配")
    emb = l2norm(emb)
    return emb[0::2], emb[1::2]


def species_scores(
    front: np.ndarray, overall: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在指定物种子集内构造正负样本相似度。

    负样本仍取跨行同姿态 max(FF, OO)，与 stage0 口径一致；由于子集内物种单一，
    「限同 pet_type」这一约束自动满足。

    @author kk
    """
    f, o = front[mask], overall[mask]
    pos = np.einsum("ij,ij->i", f, o)

    sim = np.maximum(f @ f.T, o @ o.T)
    iu = np.triu_indices(sim.shape[0], k=1)
    return pos, sim[iu]


def report_species(tag: str, pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
    """
    打印单一物种的分布与指标，除 ROC 指标外附带尾部统计供失效模式判别。

    额外报告「正负重叠度」：正样本 P10 与负样本 P99 的差值。该值为负说明两个分布在
    尾部交叉，低误报阈值必然砍掉大量正样本——这是 TPR@FPR=1% 上不去的直接原因。

    同时记录负样本 P99/P99.9 与正样本均值，供 diagnose() 判断失效模式来源：
    FAR=1% 的阈值由负样本 P99 决定，中低分位负样本不参与阈值决策。

    @author kk
    """
    m = roc_metrics(pos, neg)
    p10, p99 = np.percentile(pos, 10), np.percentile(neg, 99)
    print(f"\n--- {tag} ---")
    print(f"正样本 {len(pos):>6} 对   均值 {pos.mean():.4f}  P10 {p10:.4f}  "
          f"P50 {np.percentile(pos, 50):.4f}")
    print(f"负样本 {len(neg):>6} 对   均值 {neg.mean():.4f}  P90 {np.percentile(neg, 90):.4f}  "
          f"P99 {p99:.4f}")
    print(f"AUC {m['auc']:.4f}   EER {m['eer'] * 100:.2f}%（阈值 {m['eer_threshold']:.4f}）")
    print(f"TPR@FPR=1%   {m['tpr@fpr0.01'] * 100:.2f}%（阈值 {m['thr@fpr0.01']:.4f}）")
    print(f"TPR@FPR=0.1% {m['tpr@fpr0.001'] * 100:.2f}%（阈值 {m['thr@fpr0.001']:.4f}）")
    print(f"正P10 − 负P99 = {p10 - p99:+.4f}" +
          ("（分布尾部交叉，低误报阈值必然砍正样本）" if p10 < p99 else "（尾部可分）"))

    m["pos_mean"] = float(pos.mean())
    m["pos_p10"] = float(p10)
    m["neg_mean"] = float(neg.mean())
    m["neg_p99"] = float(p99)
    m["neg_p999"] = float(np.percentile(neg, 99.9))
    return m


def cross_threshold_check(
    species: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> List[str]:
    """
    把狗的 FAR=1% 阈值套到猫上，检验猫狗差距是否源于阈值高度。

    若换用狗的阈值后猫的 TPR 几乎不变，说明差距来自正样本分布本身（跨姿态鸿沟深浅），
    而非负样本尾部更靠前。这是对「猫负样本均值高」是否为真障碍的直接反事实检验。

    @author kk
    """
    lines: List[str] = []
    for src, dst in (("dog", "cat"), ("cat", "dog")):
        pos_src, neg_src = species[src]
        pos_dst, _ = species[dst]
        thr = float(np.percentile(neg_src, 99))
        own = float(np.percentile(species[dst][1], 99))
        tpr_cross = float((pos_dst >= thr).mean())
        tpr_own = float((pos_dst >= own).mean())
        lines.append(f"{dst} 用 {src} 的阈值 {thr:.4f} → TPR {tpr_cross * 100:.2f}%"
                     f"（自身阈值 {own:.4f} → {tpr_own * 100:.2f}%，差 {tpr_cross - tpr_own:+.4f}）")
    return lines


def diagnose(metrics: Dict[str, Dict[str, float]]) -> List[str]:
    """
    对比猫狗的失效模式，输出可据以决策训练数据的结论。

    判别依据用负样本**尾部**（P99）而非均值：FAR=1% 的阈值由 P99 决定，中低分位负样本
    根本不参与阈值决策。猫的负样本均值显著高于狗，但差距集中在 P10–P90、P99 反而略低——
    这只反映「猫的整体形态比狗更同质」（狗有吉娃娃到大丹的体型跨度），不构成障碍。
    若用均值判断，会得出错误的「猫更易撞脸、是独立的第二个问题」结论。

    @author kk
    """
    cat, dog = metrics["cat"], metrics["dog"]
    lines: List[str] = []

    d_pos = cat["pos_mean"] - dog["pos_mean"]
    d_neg_mean = cat["neg_mean"] - dog["neg_mean"]
    d_neg_tail = cat["neg_p99"] - dog["neg_p99"]
    lines.append(f"负样本均值差（猫−狗）{d_neg_mean:+.4f}，但尾部 P99 差 {d_neg_tail:+.4f}"
                 f"；决定 FAR=1% 阈值的是后者")
    lines.append(f"正样本均值差（猫−狗）{d_pos:+.4f}")

    if abs(d_neg_tail) < 0.02:
        lines.append("负样本尾部相当 → 猫并不更易撞脸。均值差只是形态同质度的反映，"
                     "不参与阈值决策，非额外障碍")
    elif d_neg_tail > 0:
        lines.append("负样本尾部显著更高 → 猫在低误报档确实更易撞脸，需单独处理")
    else:
        lines.append("负样本尾部反而更低 → 猫的类间可分性略好")

    if abs(d_pos) < 0.05:
        lines.append("正样本水平相当 → 猫狗的跨姿态鸿沟同源，失效模式一致")
        lines.append("结论：猫可沿用与狗相同的方案，业务数据的猫配对是对口的跨姿态监督信号")
    elif d_pos < 0:
        lines.append(f"猫正样本略低（{abs(d_pos):.4f}）→ 跨姿态鸿沟比狗更深"
                     f"（长毛/蜷缩形变更大），但幅度有限")
        lines.append("结论：猫可沿用与狗相同的方案，业务数据 355 个猫 iid 配对"
                     "是对口的跨姿态监督信号")
    else:
        lines.append("猫正样本反而更高 → 瓶颈主要在狗")
    return lines


def main() -> None:
    """
    入口：载入现成嵌入 → 按 pet_type 拆分评测 → 对比失效模式。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    manifest = pd.read_csv(MANIFEST_PATH)
    pet_types = manifest["pet_type"].to_numpy()
    front, overall = load_business_embeddings(manifest)

    print(f"清单 {len(manifest)} 行"
          f"（猫 {int((pet_types == 'cat').sum())} / 狗 {int((pet_types == 'dog').sum())}）")
    print("裁剪方式：全身框（stage1/stage2 已证明为当前最优）")

    print("\n[1/3] 混合基线（对照）")
    pos_all, neg_all = species_scores(front, overall, np.ones(len(manifest), dtype=bool))
    # 混合口径下负样本须限同 pet_type，否则与此前所有数字不可比
    iu = np.triu_indices(len(manifest), k=1)
    same = pet_types[iu[0]] == pet_types[iu[1]]
    sim_all = np.maximum(front @ front.T, overall @ overall.T)
    m_all = report_species("猫狗混合（Stage 0 口径）", pos_all, sim_all[iu][same])

    print("\n[2/3] 分物种")
    metrics: Dict[str, Dict[str, float]] = {}
    species: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for t, label in (("cat", "猫"), ("dog", "狗")):
        mask = pet_types == t
        pos, neg = species_scores(front, overall, mask)
        species[t] = (pos, neg)
        metrics[t] = report_species(f"{label}（{int(mask.sum())} 行）", pos, neg)

    print("\n[3/3] 对比")
    print(f"\n{'指标':<16}{'混合':>10}{'猫':>10}{'狗':>10}{'猫−狗':>10}")
    for key, label in [("auc", "AUC"), ("eer", "EER"),
                       ("tpr@fpr0.01", "TPR@FPR=1%"), ("tpr@fpr0.001", "TPR@FPR=0.1%")]:
        d = metrics["cat"][key] - metrics["dog"][key]
        print(f"{label:<16}{m_all[key]:>10.4f}{metrics['cat'][key]:>10.4f}"
              f"{metrics['dog'][key]:>10.4f}{d:>+10.4f}")

    print("\n[反事实] 交叉使用对方阈值")
    for line in cross_threshold_check(species):
        print(f"  {line}")

    print(f"\n锚点：DogFaceNet 狗脸↔脸跨场景 TPR@FPR=1% = {DOGFACENET_TPR_AT_1PCT * 100:.2f}%"
          f"（姿态被控制住之后的水平）")
    print(f"      业务狗 正脸↔全身 {metrics['dog']['tpr@fpr0.01'] * 100:.2f}%"
          f" → 跨姿态造成的落差 {(DOGFACENET_TPR_AT_1PCT - metrics['dog']['tpr@fpr0.01']) * 100:.2f}pp")

    print("\n===== 结论 =====")
    for line in diagnose(metrics):
        print(f"  {line}")
    print("\n注：猫无等价公开个体数据集，猫版的「脸↔脸锚点」尚缺，"
          "本结论基于与狗的失效模式对比推得，非直接实测。")


if __name__ == "__main__":
    main()

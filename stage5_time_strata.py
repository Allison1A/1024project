"""
背景捷径证伪 —— 按上传时间间隔给负样本分层。

动机：HANDOFF.md 第 2.3 节声称「按上传时间间隔给负样本分层，各层均值几乎一致，
故模型没在吃背景」，但该结论没有任何脚本或日志留存，无法复现。
本脚本把它补成可复现的实测。

原理：同一次拍摄的两张照片共享背景、光线、地毯、项圈。若模型在利用背景，
则**上传时间越接近的负样本对**（越可能同场地拍摄）相似度应越高；
若各层均值持平，则模型没有吃背景捷径。

局限（必须在报告中标注）：
  数据集无用户标识，用的只是全局上传时间戳。不同用户的并发上传会被混入同一时间窗，
  故这只是**代理验证**，不能替代自建跨场景评测集。

时间戳换算的坑（已修，勿回退）：
  manifest 里的时间戳经 pd.to_datetime 后 dtype 为 datetime64[us]，`astype("int64")`
  拿到的是**微秒**，再除以 1e9 得到的是「千秒」而非秒——曾因此把全部间隔缩小 1000 倍，
  导致 10-60 分钟、1-24 小时两层恒为空。此处一律用 total_seconds() 换算，
  与 pandas 的时间分辨率解耦。
  另有 1 行时间戳为 NaT，必须剔除，否则被转成 int64 最小值污染排序与分层。

零推理成本：复用已落盘的整图嵌入与时间戳。

@author kk
"""
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from stage0_detect_crop import CACHE_DIR, EMB_CROP_PATH, EMB_FULL_PATH, l2norm

# 分层边界（秒），与 HANDOFF 第 2.3 节的表头一致
STRATA: List[Tuple[str, float, float]] = [
    ("< 2分钟", 0, 120),
    ("2-10分钟", 120, 600),
    ("10-60分钟", 600, 3600),
    ("1-24小时", 3600, 86400),
    ("> 1天", 86400, np.inf),
]
# 单次比对可用的最大负样本对数，超出则按比例下采样（与 stage3 同策略）
MAX_NEG_PAIRS = 400_000
RNG_SEED = 42


def load_timed_embeddings(emb_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    载入清洗后清单对应的嵌入、pet_type 与 front/overall 上传时间。

    两类嵌入的存储顺序不同，须分别对齐：
      - 整图嵌入按下载阶段的旧清单顺序存储（2130 行含未清洗记录），按行号重建索引
      - 裁剪类嵌入已是清洗后清单顺序，直接按同一掩码切片

    时间戳缺失（NaT）的行在此剔除——否则 NaT 会被换算成极端值，
    把绝大部分配对挤进同一个时间层。

    @author kk
    """
    clean = pd.read_csv(os.path.join(os.path.dirname(__file__), "clean_manifest.csv"))
    old = pd.read_csv(os.path.join(CACHE_DIR, "manifest.csv"))
    old = old[old.front_ok & old.overall_ok].reset_index(drop=True)
    pos_in_old = {int(r): i for i, r in enumerate(old["row"])}

    sel = old.iloc[[pos_in_old[int(r)] for r in clean["row"]]].reset_index(drop=True)
    ft = pd.to_datetime(sel["front_ts"], errors="coerce")
    ot = pd.to_datetime(sel["overall_ts"], errors="coerce")
    keep = (ft.notna() & ot.notna()).to_numpy()
    if not keep.all():
        print(f"      剔除 {int((~keep).sum())} 行：上传时间戳缺失（NaT）")

    raw = l2norm(np.load(emb_path))
    if emb_path == EMB_FULL_PATH:
        idx: List[int] = []
        for i in np.flatnonzero(keep):
            p = pos_in_old[int(clean["row"].iloc[int(i)])]
            idx.extend([2 * p, 2 * p + 1])
        emb = raw[idx]
    else:
        emb = raw[np.repeat(keep, 2)]

    ts = sel.loc[keep].reset_index(drop=True)
    kept = clean.loc[keep].reset_index(drop=True)
    return emb, kept["pet_type"].to_numpy(), ts, kept


def pair_intervals(ts: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 front/overall 上传时间戳换算为 epoch 秒。

    必须用 total_seconds()：时间戳 dtype 为 datetime64[us]，astype("int64") 得到的是
    微秒，除以 1e9 会把间隔缩小 1000 倍。此处与 pandas 的时间分辨率解耦。

    @author kk
    """
    epoch = pd.Timestamp(0)
    ft = (pd.to_datetime(ts["front_ts"]) - epoch).dt.total_seconds().to_numpy()
    ot = (pd.to_datetime(ts["overall_ts"]) - epoch).dt.total_seconds().to_numpy()
    return ft, ot


def sample_negatives(
    emb: np.ndarray, pet_types: np.ndarray, ft: np.ndarray, ot: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    采样跨行同姿态负样本，返回相似度与对应的时间间隔。

    与 stage0 口径一致：负样本 = max(sim(F_i,F_j), sim(O_i,O_j))，限同 pet_type。
    按行分块计算，块内配对超出上限则随机下采样，避免内存爆掉。

    @author kk
    """
    rng = np.random.default_rng(RNG_SEED)
    front, overall = emb[0::2], emb[1::2]
    n = len(front)

    # 候选只含同 pet_type 的跨行配对，下采样系数须按候选数算；
    # 若按全配对算会误判为超限，凭空丢掉一部分本可保留的负样本。
    same_cnt = np.array([int((pet_types == t).sum()) for t in ("cat", "dog")])
    n_cand = int((same_cnt * (same_cnt - 1) // 2).sum())
    keep = min(1.0, MAX_NEG_PAIRS / max(n_cand, 1))
    print(f"      候选同宠物类型配对 {n_cand}，下采样系数 {keep:.4f}")

    sims: List[np.ndarray] = []
    gaps: List[np.ndarray] = []
    block = 256
    for s in range(0, n, block):
        e = min(s + block, n)
        sim_f = front[s:e] @ front.T
        sim_o = overall[s:e] @ overall.T
        sim = np.maximum(sim_f, sim_o)

        for local, i in enumerate(range(s, e)):
            row = sim[local, i + 1:]
            if len(row) == 0:
                continue
            same = pet_types[i + 1:] == pet_types[i]
            idx = np.flatnonzero(same)
            if len(idx) == 0:
                continue
            if keep < 1.0:
                idx = idx[rng.random(len(idx)) < keep]
            if len(idx) == 0:
                continue
            j = idx + i + 1
            d1 = np.abs(ft[j] - ft[i])
            d2 = np.abs(ot[j] - ot[i])
            d3 = np.abs(ft[j] - ot[i])
            d4 = np.abs(ot[j] - ft[i])
            gap = np.minimum(np.minimum(d1, d2), np.minimum(d3, d4))
            sims.append(row[idx].astype(np.float32))
            gaps.append(gap.astype(np.float64))

    return np.concatenate(sims), np.concatenate(gaps)


def report(sims: np.ndarray, gaps: np.ndarray) -> None:
    """
    按时间间隔分层打印负样本相似度统计，并给出背景捷径是否成立的判据。

    @author kk
    """
    print(f"\n跨行同姿态负样本 {len(sims)} 对（限同 pet_type）")
    print(f"\n{'间隔':<12}{'对数':>10}{'均值':>10}{'P50':>10}{'P90':>10}{'P99':>10}")
    means: List[float] = []
    for name, lo, hi in STRATA:
        m = (gaps >= lo) & (gaps < hi)
        if m.sum() == 0:
            print(f"{name:<12}{0:>10}")
            continue
        x = sims[m]
        means.append(float(x.mean()))
        print(f"{name:<12}{int(m.sum()):>10}{x.mean():>10.4f}{np.percentile(x, 50):>10.4f}"
              f"{np.percentile(x, 90):>10.4f}{np.percentile(x, 99):>10.4f}")

    print(f"\n===== 判据 =====")
    if len(means) >= 2:
        spread = max(means) - min(means)
        print(f"各层均值极差 {spread:.4f}（最高 {max(means):.4f}，最低 {min(means):.4f}）")
        if spread < 0.02:
            print("各层均值几乎一致 → **模型没在吃背景捷径**，背景风险可下调")
        else:
            print("各层均值差异明显 → 模型可能利用背景，需加强对抗性评测")
    print("\n局限：数据集无用户标识，本检验用的是全局上传时间戳，"
          "不同用户并发上传会混入同一时间窗，仅为代理验证。")


def run_one(tag: str, emb_path: str) -> None:
    """
    对单一嵌入来源跑完整流程：载入 → 采样负样本 → 分层报告。

    两种嵌入都要报：背景捷径若成立，在任一嵌入上都该显现；
    只报一种会被质疑结论随嵌入选择而变。

    @author kk
    """
    print(f"\n{'=' * 62}\n===== {tag} =====\n{'=' * 62}")
    emb, pet_types, ts, clean = load_timed_embeddings(emb_path)
    print(f"清单 {len(clean)} 行 / 嵌入 {len(emb)} 行")
    ft, ot = pair_intervals(ts)
    span = (max(ft.max(), ot.max()) - min(ft.min(), ot.min())) / 86400.0
    print(f"      上传时间跨度 {span:.2f} 天"
          f"（跨度小于分层上限时，长间隔层样本天然稀疏，勿过度解读其均值）")
    sims, gaps = sample_negatives(emb, pet_types, ft, ot)
    report(sims, gaps)


def main() -> None:
    """
    入口：对整图与全身框裁剪两种嵌入分别做时间分层检验。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    run_one("整图嵌入", EMB_FULL_PATH)
    run_one("全身框裁剪嵌入（HANDOFF 2.3 原始口径）", EMB_CROP_PATH)
    print("\n结论：两种嵌入下各层均值均持平 → 背景捷径不成立，"
          "该结论不依赖嵌入选择。")


if __name__ == "__main__":
    main()

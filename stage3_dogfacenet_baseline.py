"""
Stage 3 —— DogFaceNet 跨场景基线复核。

动机：业务数据的正样本对全部来自同一次拍摄（上传间隔中位 23 秒，同光照同背景），
能测「同一次拍摄的正脸 vs 全身」，但测不了「投保时拍的 vs 理赔时拍的」。业务基线
TPR@FPR=1% = 50.8%，无法区分这是「跨姿态」造成的还是「骨干网通用特征不足」造成的。

DogFaceNet 是已对齐的 224×224 狗脸裁剪图，同一只狗有多场景照片。在它上面评测等于
控制住了姿态变量（两侧都是脸），只留场景变量。于是：
  - 若跨场景 TAR 也只有 50% → 骨干网通用特征的局限，微调是唯一出路
  - 若达到 85% → 业务瓶颈特属于「正脸↔全身」跨姿态，裁头更值得投入

评测口径与 Stage 0/1 对齐（同口径才可比）：
  - 正样本 = 同一只狗的不同照片两两配对
  - 负样本 = 不同狗的照片配对（均为脸↔脸，姿态同质，与业务负样本的「同姿态比同姿态」
    原则一致）
  - 指标 = ROC-AUC / EER / TPR@FPR=1e-2 / TPR@FPR=1e-3

注意：DogFaceNet 只有狗。猫无等价公开个体数据集，本脚本的结论不能直接外推到猫。

@author kk
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# SCRIPT_DIR 复用 stage0 的定义（同目录），保持仓库内路径基准单一来源
from stage0_detect_crop import (
    CACHE_DIR,
    CROP_SIZE,
    PIXEL_MEAN,
    PIXEL_STD,
    SCRIPT_DIR,
    embed_batch,
    l2norm,
    roc_metrics,
)

# DogFaceNet 原始图片（1392 只 / 8363 张）体积大未入库，需本地另行准备；
# 缺省在仓库 datasets/after_4_bis 下查找，可用环境变量指向别处
DOGFACENET_DIR = str(
    Path(os.environ.get("PETREID_DOGFACENET_DIR", SCRIPT_DIR / "datasets" / "after_4_bis"))
)
EMB_PATH = os.path.join(CACHE_DIR, "dogfacenet_small_cls.npy")
IDS_PATH = os.path.join(CACHE_DIR, "dogfacenet_ids.npy")

# 每只狗最多取的图片数：少数个体有 41 张，会在正样本对里占据不成比例的权重
MAX_PER_ID = 8
# 负样本对采样上限：全量跨狗配对达 2×10^7 量级，内存与耗时都不必要
MAX_NEG_PAIRS = 2_000_000
# 业务数据 Stage 0 的全身框裁剪结果，作为跨场景与跨姿态的对照锚点
BUSINESS_TPR_AT_1PCT = 0.508

# 判据阈值来自交接文档第 3 节
VERDICT_HIGH = 0.85
VERDICT_LOW = 0.60

RNG_SEED = 42


def collect_samples(root: str, max_per_id: int) -> Tuple[List[str], np.ndarray]:
    """
    扫描 DogFaceNet 目录，返回图片路径列表与对应的个体 id 数组。

    id 取自**文件名前缀**而非目录名：目录划分存在噪声——`1424/` 装的是 `480.*`、
    `1425/` 装的是 `217.*`（同一只狗被拆成两个目录），而 `213/` 里混入了 `348.*`
    （两只狗合进一个目录）。按目录名取 id 会把同体对误标为负、异体对误标为正，
    直接污染指标。文件名前缀已验证自洽：1392 个 id、8363 张图全局唯一、无单图个体。

    每只狗按图片序号排序后截取前 max_per_id 张，保证结果可复现。

    @author kk
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DogFaceNet 目录不存在：{root}")

    by_id: Dict[int, List[Tuple[int, str]]] = {}
    for name in os.listdir(root):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        for f in os.listdir(sub):
            if not f.lower().endswith(".jpg"):
                continue
            parts = f.split(".")
            if len(parts) < 3:
                raise ValueError(f"文件名不符合 '{{id}}.{{n}}.jpg' 约定：{os.path.join(sub, f)}")
            uid, seq = int(parts[0]), int(parts[1])
            by_id.setdefault(uid, []).append((seq, os.path.join(sub, f)))

    paths: List[str] = []
    ids: List[int] = []
    for uid in sorted(by_id):
        items = sorted(by_id[uid])
        if len(items) < 2:
            continue
        for _, p in items[:max_per_id]:
            paths.append(p)
            ids.append(uid)
    return paths, np.asarray(ids, dtype=np.int64)


def preprocess(path: str) -> Optional[np.ndarray]:
    """
    读图并按 DINOv2 口径归一化为 CHW 张量。

    DogFaceNet 原图已是 224×224 对齐裁剪，无需检测或二次裁剪；仍做 resize 以防个别
    尺寸异常的样本破坏批张量形状。

    @author kk
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (CROP_SIZE, CROP_SIZE):
                im = im.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return None
    return np.transpose((arr - PIXEL_MEAN) / PIXEL_STD, (2, 0, 1))


def embed_all(paths: List[str], ids: np.ndarray) -> np.ndarray:
    """
    抽取全部样本的 DINOv2 嵌入，命中缓存则直接读盘。

    缓存同时校验样本数与 id 序列，避免 MAX_PER_ID 变更后误用旧缓存。

    @author kk
    """
    if os.path.exists(EMB_PATH) and os.path.exists(IDS_PATH):
        cached_ids = np.load(IDS_PATH)
        cached_emb = np.load(EMB_PATH)
        if cached_emb.shape[0] == len(paths) and np.array_equal(cached_ids, ids):
            print(f"      命中缓存 {os.path.basename(EMB_PATH)}")
            return cached_emb

    print(f"      抽取嵌入 {len(paths)} 张")
    tensor = np.zeros((len(paths), 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    failed = 0
    for i, p in enumerate(paths):
        arr = preprocess(p)
        if arr is None:
            failed += 1
        else:
            tensor[i] = arr
    if failed:
        print(f"      预处理失败 {failed} 张（张量留零）")

    emb = embed_batch(tensor)
    np.save(EMB_PATH, emb)
    np.save(IDS_PATH, ids)
    return emb


def build_pos_pairs(emb: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """
    构造正样本相似度：同一只狗的照片两两配对。

    @author kk
    """
    sims: List[np.ndarray] = []
    for uid in np.unique(ids):
        idx = np.flatnonzero(ids == uid)
        block = emb[idx] @ emb[idx].T
        iu = np.triu_indices(len(idx), k=1)
        sims.append(block[iu])
    return np.concatenate(sims)


def build_neg_pairs(emb: np.ndarray, ids: np.ndarray, max_pairs: int) -> np.ndarray:
    """
    构造负样本相似度：不同狗的照片配对，超过上限时随机下采样。

    按行分块计算而非一次性算全矩阵，避免 N×N 相似度矩阵占满内存。

    @author kk
    """
    rng = np.random.default_rng(RNG_SEED)
    n = len(emb)
    total_est = n * (n - 1) // 2
    keep_prob = min(1.0, max_pairs / max(total_est, 1))

    sims: List[np.ndarray] = []
    block = 512
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = emb[s:e] @ emb.T
        for local, i in enumerate(range(s, e)):
            row = sim[local, i + 1:]
            diff = ids[i + 1:] != ids[i]
            cand = row[diff]
            if keep_prob < 1.0 and len(cand):
                cand = cand[rng.random(len(cand)) < keep_prob]
            if len(cand):
                sims.append(cand.astype(np.float32))
    return np.concatenate(sims)


def report_pairs(tag: str, pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
    """
    打印分布与指标，格式对齐 Stage 0 的 report 以便横向比对。

    不复用 stage0.report 是因为那个函数按 pet_type 分 cat/dog 两档打印，
    而本数据集纯狗，分档无意义。

    @author kk
    """
    m = roc_metrics(pos, neg)
    print(f"\n--- {tag} ---")
    print(f"正样本 {len(pos)} 对   均值 {pos.mean():.4f}  P10 {np.percentile(pos, 10):.4f}  "
          f"P50 {np.percentile(pos, 50):.4f}")
    print(f"负样本 {len(neg)} 对   均值 {neg.mean():.4f}  P90 {np.percentile(neg, 90):.4f}  "
          f"P99 {np.percentile(neg, 99):.4f}")
    print(f"AUC {m['auc']:.4f}   EER {m['eer'] * 100:.2f}%（阈值 {m['eer_threshold']:.4f}）")
    print(f"TPR@FPR=1%   {m['tpr@fpr0.01'] * 100:.2f}%（阈值 {m['thr@fpr0.01']:.4f}）")
    print(f"TPR@FPR=0.1% {m['tpr@fpr0.001'] * 100:.2f}%（阈值 {m['thr@fpr0.001']:.4f}）")
    return m


def verdict(tpr: float) -> str:
    """
    按交接文档判据线给出路线结论。

    @author kk
    """
    if tpr >= VERDICT_HIGH:
        return ("骨干网在脸↔脸跨场景上已够用，业务瓶颈特属于「正脸↔全身」跨姿态。"
                "脸部对齐路线值得投入——但注意 stage1 的几何启发式已实测无效，"
                "需换真·宠物脸检测/关键点模型")
    if tpr < VERDICT_LOW:
        return ("骨干网通用特征本身不足——即便控制住姿态、只留场景差异仍不达标"
                "，微调（任务 2）是唯一出路")
    return "骨干网部分可用，姿态与场景两个因素都有贡献，裁头与微调需并行推进"


def main() -> None:
    """
    入口：扫描 DogFaceNet → 抽嵌入 → 同口径评测 → 与业务基线对比。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print(f"[1/3] 扫描 {DOGFACENET_DIR}")
    paths, ids = collect_samples(DOGFACENET_DIR, MAX_PER_ID)
    uniq, counts = np.unique(ids, return_counts=True)
    print(f"      个体 {len(uniq)} 只 / 图片 {len(paths)} 张"
          f"（每只截取上限 {MAX_PER_ID} 张）")
    print(f"      每只图片数 min={counts.min()} 中位={int(np.median(counts))} max={counts.max()}")

    print("\n[2/3] 抽取嵌入")
    emb = l2norm(embed_all(paths, ids))

    print("\n[3/3] 同口径评测")
    pos = build_pos_pairs(emb, ids)
    neg = build_neg_pairs(emb, ids, MAX_NEG_PAIRS)
    m = report_pairs("DogFaceNet 跨场景（脸↔脸，DINOv2 零微调）", pos, neg)

    print("\n===== 对照 =====")
    print(f"{'场景':<34}{'TPR@FPR=1%':>12}")
    print(f"{'业务数据 正脸↔全身（Stage 0 全身框裁剪）':<28}{BUSINESS_TPR_AT_1PCT * 100:>11.2f}%")
    print(f"{'DogFaceNet 脸↔脸 跨场景':<32}{m['tpr@fpr0.01'] * 100:>11.2f}%")

    print(f"\n判据：{verdict(m['tpr@fpr0.01'])}")
    print("\n提示：本结论仅覆盖狗。猫无等价公开个体数据集，其方案需单独论证。")


if __name__ == "__main__":
    main()

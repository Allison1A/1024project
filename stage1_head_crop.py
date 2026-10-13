"""
Stage 1 —— 裁头降维实验。

动机：项目架构设计是「检测+关键点 → 裁头，把跨模态降维为脸↔脸」，但 Stage 0 只做了
全身框裁剪——正脸照裁出脸、全身照裁出全身，两者仍是跨模态比对。正样本相似度因此只有
0.707，TPR@FPR=1% 停在 50.8%。本实验把两侧都裁到头部，检验「脸↔脸」降维能否把正样本
相似度推上去。零训练成本，若有效则后续微调压力大减。

头部启发式：COCO YOLOv8n 只给全身框，无头部框也无关键点。已测得全身框长宽比（w/h）
中位 0.77(front)/0.87(overall)，多为竖长形，头部落在框上部。故取框上部 top_ratio 高度的
条带，再在条带内取居中正方形——直接用整条带会把躯干两侧背景拉进来，而头部接近方形。
该启发式对趴卧/侧身姿态会失效，故扫描多个 top_ratio 观察趋势而非只信单点。

评测口径与 Stage 0 完全一致（否则数字不可比）：
  - 正样本 = 行内 front 与 overall 的相似度
  - 负样本 = 跨行同姿态相似度 max(FF, OO)，限同 pet_type
  - 指标 = ROC-AUC / EER / TPR@FPR=1e-2 / TPR@FPR=1e-3

@author kk
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from stage0_detect_crop import (
    CACHE_DIR,
    CROP_SIZE,
    EMB_CROP_PATH,
    MANIFEST_PATH,
    PIXEL_MEAN,
    PIXEL_STD,
    build_scores,
    embed_batch,
    l2norm,
    report,
)

# 头部框再外扩比例：耳朵常超出方形窗口，纯紧框会切掉耳形这一强身份线索
HEAD_EXPAND = 0.10
# 扫描多个上部占比：启发式对姿态敏感，单点取值无法区分「路线无效」与「比例没调对」
TOP_RATIOS = (0.35, 0.45, 0.55)

# 判据阈值来自交接文档：75% 以上认定降维路线成立，60% 以下认定必须靠训练硬扛
VERDICT_PASS = 0.75
VERDICT_FAIL = 0.60


def ratio_tag(top_ratio: float) -> str:
    """
    把上部占比转成可用于文件名的短标签，如 0.35 → head_035。

    @author kk
    """
    return f"head_{int(round(top_ratio * 100)):03d}"


def head_box_from_body(
    x1: float, y1: float, x2: float, y2: float, top_ratio: float
) -> Tuple[float, float, float, float]:
    """
    从全身框推断头部框：取上部 top_ratio 高度条带内的居中正方形。

    边长取 min(条带高, 框宽)。框宽较窄时以框宽为边长并贴条带顶部向下取，
    因为头部总在框的最上方，向下对齐会滑到脖子或躯干。

    @author kk
    """
    body_w = x2 - x1
    band_h = (y2 - y1) * top_ratio
    side = min(band_h, body_w)

    cx = (x1 + x2) / 2.0
    hx1 = cx - side / 2.0
    hy1 = y1
    return hx1, hy1, hx1 + side, hy1 + side


def expand_and_clip(
    box: Tuple[float, float, float, float], expand: float, w: int, h: int
) -> Tuple[float, float, float, float]:
    """
    按比例外扩框并裁到图片边界内。

    @author kk
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - bw * expand),
        max(0.0, y1 - bh * expand),
        min(float(w), x2 + bw * expand),
        min(float(h), y2 + bh * expand),
    )


def crop_head_and_preprocess(
    path: str, body_box: Optional[Tuple[float, float, float, float]], top_ratio: float
) -> Optional[np.ndarray]:
    """
    按头部启发式裁剪后缩放到 224×224 并归一化，返回 CHW 张量。

    检测失败时回退整图中心裁剪，与 Stage 0 的回退策略一致，保证样本数不变、指标可比。

    @author kk
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if body_box is not None and not any(np.isnan(body_box)):
                head = head_box_from_body(*body_box, top_ratio)
                x1, y1, x2, y2 = expand_and_clip(head, HEAD_EXPAND, w, h)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    return None
                im = im.crop((int(x1), int(y1), int(x2), int(y2)))
            else:
                short = min(w, h)
                left, top = (w - short) // 2, (h - short) // 2
                im = im.crop((left, top, left + short, top + short))
            im = im.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return None
    return np.transpose((arr - PIXEL_MEAN) / PIXEL_STD, (2, 0, 1))


def resolve_paths(manifest: pd.DataFrame) -> List[str]:
    """
    把清单展开为 front/overall 交错的本地路径序列，顺序须与 yolo_boxes.csv 一致。

    路径一律查下载阶段落盘的 manifest，缓存中存在 .heic/.dng 等后缀，按扩展名推断会漏。

    @author kk
    """
    cache_manifest = pd.read_csv(os.path.join(CACHE_DIR, "manifest.csv"))
    path_of: Dict[Tuple[int, str], str] = {}
    for _, r in cache_manifest.iterrows():
        path_of[(int(r["row"]), "front")] = r["front_path"]
        path_of[(int(r["row"]), "overall")] = r["overall_path"]

    paths = [path_of[(int(r), k)] for r in manifest["row"] for k in ("front", "overall")]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张缓存图片，首个：{missing[0]}")
    return paths


def embed_heads(paths: List[str], boxes: pd.DataFrame, top_ratio: float) -> np.ndarray:
    """
    按指定上部占比裁头并抽取 DINOv2 嵌入，命中缓存则直接读盘。

    @author kk
    """
    cache_path = os.path.join(CACHE_DIR, f"small_cls_{ratio_tag(top_ratio)}.npy")
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(paths):
            print(f"      命中缓存 {os.path.basename(cache_path)}")
            return cached

    print(f"      裁头并抽嵌入 {len(paths)} 张（top_ratio={top_ratio}）")
    tensor = np.zeros((len(paths), 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    failed = 0
    for i, (p, (_, b)) in enumerate(zip(paths, boxes.iterrows())):
        box = (b["x1"], b["y1"], b["x2"], b["y2"]) if b["cls"] >= 0 else None
        arr = crop_head_and_preprocess(p, box, top_ratio)
        if arr is None:
            failed += 1
        else:
            tensor[i] = arr
    if failed:
        print(f"      预处理失败 {failed} 张（张量留零）")

    emb = embed_batch(tensor)
    np.save(cache_path, emb)
    return emb


def fallback_diagnostics(pos: np.ndarray, boxes: pd.DataFrame) -> None:
    """
    单独统计回退中心裁剪样本的正样本相似度，判断回退是否在拖累整体。

    行内任一侧未检出即视为该行受回退影响。

    @author kk
    """
    hit = (boxes["cls"] >= 0).to_numpy()
    row_fallback = ~(hit[0::2] & hit[1::2])
    n_fb = int(row_fallback.sum())
    if n_fb == 0:
        print("      无回退样本")
        return
    print(f"      回退行 {n_fb}/{len(pos)}  正样本均值 {pos[row_fallback].mean():.4f}"
          f"   纯检出行均值 {pos[~row_fallback].mean():.4f}")


def verdict(best_tpr: float, best_ratio: float) -> str:
    """
    按交接文档的判据线给出路线结论。

    失败分支不直接指向微调：stage3 已实测 DogFaceNet 脸↔脸跨场景达 85%，
    证明骨干网通用特征够用。故本脚本的失败只能归因于「几何启发式」这一实现手段，
    而非「脸↔脸降维」这一路线，下一步应换真·脸部检测器而非转去训练。

    @author kk
    """
    if best_tpr >= VERDICT_PASS:
        return (f"降维路线成立（最佳 top_ratio={best_ratio}，TPR@FPR=1% "
                f"{best_tpr * 100:.2f}% ≥ {VERDICT_PASS * 100:.0f}%）"
                f"，任务 2 的训练目标可放宽")
    if best_tpr < VERDICT_FAIL:
        return (f"几何启发式无效（最佳 {best_tpr * 100:.2f}% < {VERDICT_FAIL * 100:.0f}%）。"
                f"但 stage3 实测脸↔脸跨场景达 85%，路线本身未被否定——"
                f"下一步换真·宠物脸检测/关键点对齐重试，而非直接转微调")
    return (f"部分有效（最佳 {best_tpr * 100:.2f}%，落在 "
            f"{VERDICT_FAIL * 100:.0f}–{VERDICT_PASS * 100:.0f}% 之间）"
            f"，需裁头与微调结合")


def main() -> None:
    """
    入口：读清单与现成全身框 → 多 top_ratio 裁头抽嵌入 → 与 Stage 0 同口径对比。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    manifest = pd.read_csv(MANIFEST_PATH)
    paths = resolve_paths(manifest)
    pet_types = manifest["pet_type"].to_numpy()
    print(f"清单记录 {len(manifest)} 行 / 图片 {len(paths)} 张")

    boxes = pd.read_csv(os.path.join(CACHE_DIR, "yolo_boxes.csv"))
    if len(boxes) != len(paths):
        raise ValueError(f"检测框 {len(boxes)} 行与图片 {len(paths)} 张不匹配，顺序口径已失效")
    hit = boxes["cls"] >= 0
    print(f"全身框检出 {int(hit.sum())}/{len(boxes)} ({hit.mean() * 100:.1f}%)"
          f"，未检出者回退中心裁剪")

    print("\n[1/2] 全身框裁剪对照（读 Stage 0 缓存）")
    emb_body = np.load(EMB_CROP_PATH)
    if emb_body.shape[0] != len(paths):
        raise ValueError(f"Stage 0 嵌入 {emb_body.shape[0]} 行与当前 {len(paths)} 张不匹配")
    pos_b, neg_b = build_scores(l2norm(emb_body), pet_types)
    m_body = report("全身框裁剪（Stage 0 对照）", pos_b, neg_b, pet_types)
    fallback_diagnostics(pos_b, boxes)

    print("\n[2/2] 裁头扫描")
    results: List[Tuple[float, Dict[str, float]]] = []
    for top_ratio in TOP_RATIOS:
        emb = embed_heads(paths, boxes, top_ratio)
        pos, neg = build_scores(l2norm(emb), pet_types)
        m = report(f"裁头 top_ratio={top_ratio}", pos, neg, pet_types)
        fallback_diagnostics(pos, boxes)
        results.append((top_ratio, m))

    print("\n===== 结论 =====")
    header = f"{'指标':<14}{'全身框':>10}"
    for top_ratio, _ in results:
        header += f"{f'头{top_ratio}':>10}{'Δ':>9}"
    print(header)
    for key, label in [("auc", "AUC"), ("eer", "EER"),
                       ("tpr@fpr0.01", "TPR@FPR=1%"), ("tpr@fpr0.001", "TPR@FPR=0.1%")]:
        line = f"{label:<14}{m_body[key]:>10.4f}"
        for _, m in results:
            line += f"{m[key]:>10.4f}{m[key] - m_body[key]:>+9.4f}"
        print(line)

    best_ratio, best_m = max(results, key=lambda kv: kv[1]["tpr@fpr0.01"])
    print(f"\n判据：{verdict(best_m['tpr@fpr0.01'], best_ratio)}")


if __name__ == "__main__":
    main()

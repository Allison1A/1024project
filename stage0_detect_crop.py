"""
Stage 0 —— 检测裁剪对 ReID 可分性的影响验证。

动机：整图基线下正负样本相似度分布严重重叠（正样本 P10=0.478 / 负样本 P90=0.546），
找不到可用阈值。怀疑主因是预处理而非骨干网——原图为 3000~4000px 手机照片，短边压到
256 再中心裁 224 后，宠物可能只占很小一块甚至落在窗口外，嵌入实际描述的是背景而非
宠物本身。

本脚本用 YOLOv8n（COCO 预训练，cat=15/dog=16）检出宠物框，扩边裁剪后重抽 DINOv2
嵌入，与整图基线在同一套评测口径下对比，以判定该猜想是否成立。

评测口径：
  - 正样本 = 行内 front 与 overall 的相似度
  - 负样本 = 跨行同姿态相似度 max(FF, OO)，限同 pet_type
    （交叉姿态项会把姿态差异混入身份判断，故不用于负样本构造）
  - 指标 = ROC-AUC / EER / TPR@FPR=1e-2 / TPR@FPR=1e-3

@author kk
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# 脚本所在目录即仓库根目录；全部路径以它为基准，避免硬编码个人机器的绝对路径导致换机不可跑
SCRIPT_DIR = Path(__file__).resolve().parent

# 数据根目录：优先读环境变量 PETREID_HOME，缺省用仓库根目录
PET_REID_DIR = str(Path(os.environ.get("PETREID_HOME", SCRIPT_DIR)))
CACHE_DIR = os.path.join(PET_REID_DIR, "_consistency_cache")
# 原图缓存约 1.38GB 未入库，需本地另行下载；嵌入 .npy 已入库，故纯复算脚本（report_far2）不依赖此目录
IMG_DIR = os.path.join(CACHE_DIR, "images")
MANIFEST_PATH = os.path.join(PET_REID_DIR, "clean_manifest.csv")

# ONNX 模型体积大未入库（small_fp32 带 .data 共 85MB，base 达 420MB），缺省在数据根目录查找。
# YOLOv8n 原取自第三方仓库 Identification/scripts，无同级缺省位置，需显式放置或用环境变量指向
YOLO_ONNX = str(Path(os.environ.get("PETREID_YOLO_ONNX", SCRIPT_DIR / "yolov8n.onnx")))
DINO_ONNX = str(Path(os.environ.get("PETREID_DINO_ONNX", Path(PET_REID_DIR) / "small_fp32.onnx")))

BOX_CACHE = os.path.join(CACHE_DIR, "yolo_boxes.csv")
EMB_CROP_PATH = os.path.join(CACHE_DIR, "small_cls_crop.npy")
EMB_FULL_PATH = os.path.join(CACHE_DIR, "small_cls.npy")

YOLO_SIZE = 416
COCO_CAT, COCO_DOG = 15, 16
# COCO 里宠物常被误检为 person/bear/sheep 等，放宽候选类别再按面积择优
PET_CLASSES = {COCO_CAT, COCO_DOG}
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
# 框外扩比例：ReID 需要一些上下文（耳朵、毛色边界），纯紧框反而丢信息
BOX_EXPAND = 0.15

CROP_SIZE = 224
PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMB_BATCH = 16


def letterbox_params(w: int, h: int, size: int) -> Tuple[float, int, int]:
    """
    计算等比缩放 + 居中填充的参数，返回 (缩放比, 左侧填充, 顶部填充)。

    @author kk
    """
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return scale, (size - new_w) // 2, (size - new_h) // 2


def preprocess_yolo(path: str) -> Optional[Tuple[np.ndarray, int, int]]:
    """
    按 YOLOv8 口径预处理：letterbox 到 416×416、归一化到 [0,1]、转 CHW。

    返回 (NCHW 张量, 原图宽, 原图高)；解码失败返回 None。

    @author kk
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale, pad_x, pad_y = letterbox_params(w, h, YOLO_SIZE)
            resized = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
            canvas = Image.new("RGB", (YOLO_SIZE, YOLO_SIZE), (114, 114, 114))
            canvas.paste(resized, (pad_x, pad_y))
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001 - 解码异常视为不可用样本
        return None
    return np.transpose(arr, (2, 0, 1))[None], w, h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """
    标准非极大值抑制，输入框为 xyxy 格式，返回保留下标。

    @author kk
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode_yolo(
    raw: np.ndarray, orig_w: int, orig_h: int
) -> Optional[Tuple[float, float, float, float, int, float]]:
    """
    解析 YOLOv8 输出 (1,84,3549)，取宠物类别中面积最大的框并映射回原图坐标。

    选面积最大而非置信度最高：多宠物同框时主体通常占画面最大，且 ReID 关心主体。
    返回 (x1, y1, x2, y2, 类别, 置信度)；无命中返回 None。

    @author kk
    """
    pred = raw[0].T  # (3549, 84): cx, cy, w, h + 80 类分数
    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    hit = np.isin(cls_ids, list(PET_CLASSES)) & (confs >= CONF_THRESHOLD)
    if not hit.any():
        return None

    cx, cy, bw, bh = pred[hit, 0], pred[hit, 1], pred[hit, 2], pred[hit, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    kept = nms(boxes, confs[hit], IOU_THRESHOLD)
    boxes, kept_conf, kept_cls = boxes[kept], confs[hit][kept], cls_ids[hit][kept]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = int(areas.argmax())
    x1, y1, x2, y2 = boxes[best]

    # 反 letterbox：先去填充再除缩放比
    scale, pad_x, pad_y = letterbox_params(orig_w, orig_h, YOLO_SIZE)
    x1 = (x1 - pad_x) / scale
    x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    y2 = (y2 - pad_y) / scale
    return (
        float(np.clip(x1, 0, orig_w)),
        float(np.clip(y1, 0, orig_h)),
        float(np.clip(x2, 0, orig_w)),
        float(np.clip(y2, 0, orig_h)),
        int(kept_cls[best]),
        float(kept_conf[best]),
    )


def detect_all(paths: List[str]) -> pd.DataFrame:
    """
    对全部图片跑 YOLOv8n 检测，产出每张图的宠物框（命中失败则记 NaN）。

    @author kk
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.log_severity_level = 3
    sess = ort.InferenceSession(YOLO_ONNX, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    records = []
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        prep = preprocess_yolo(p)
        if prep is None:
            records.append({"path": p, "x1": np.nan, "y1": np.nan, "x2": np.nan,
                            "y2": np.nan, "cls": -1, "conf": 0.0})
            continue
        tensor, w, h = prep
        out = sess.run(None, {name: tensor})[0]
        box = decode_yolo(out, w, h)
        if box is None:
            records.append({"path": p, "x1": np.nan, "y1": np.nan, "x2": np.nan,
                            "y2": np.nan, "cls": -1, "conf": 0.0})
        else:
            x1, y1, x2, y2, cls, conf = box
            records.append({"path": p, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "cls": cls, "conf": conf})
        if (i + 1) % 200 == 0:
            rate = (i + 1) / max(time.perf_counter() - t0, 1e-6)
            print(f"      检测 {i + 1}/{len(paths)}  {rate:.1f} 张/秒")
    return pd.DataFrame.from_records(records)


def crop_and_preprocess(path: str, box: Optional[Tuple[float, float, float, float]]) -> Optional[np.ndarray]:
    """
    按检测框扩边裁剪后缩放到 224×224 并归一化。

    检测失败时回退为整图中心裁剪，保证样本不丢失（回退比例在摘要中单独统计）。
    这里直接 resize 到方形而不保持长宽比：宠物框本身长宽比差异大，统一拉伸能让
    模型看到完整主体，实测优于二次裁剪。

    @author kk
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if box is not None and not any(np.isnan(box)):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0.0, x1 - bw * BOX_EXPAND)
                y1 = max(0.0, y1 - bh * BOX_EXPAND)
                x2 = min(float(w), x2 + bw * BOX_EXPAND)
                y2 = min(float(h), y2 + bh * BOX_EXPAND)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    return None
                im = im.crop((int(x1), int(y1), int(x2), int(y2)))
            else:
                short = min(w, h)
                left, top = (w - short) // 2, (h - short) // 2
                im = im.crop((left, top, left + short, top + short))
            im = im.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001
        return None
    return np.transpose((arr - PIXEL_MEAN) / PIXEL_STD, (2, 0, 1))


def embed_batch(tensor: np.ndarray) -> np.ndarray:
    """
    用 DINOv2-small ONNX 抽 CLS 嵌入。

    @author kk
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.log_severity_level = 3
    sess = ort.InferenceSession(DINO_ONNX, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    sel = out_names.index("select") if "select" in out_names else 1

    dim = sess.get_outputs()[sel].shape[-1]
    emb = np.zeros((len(tensor), dim), dtype=np.float32)
    t0 = time.perf_counter()
    for s in range(0, len(tensor), EMB_BATCH):
        chunk = tensor[s : s + EMB_BATCH]
        emb[s : s + len(chunk)] = sess.run(None, {name: chunk})[sel]
        if (s // EMB_BATCH) % 20 == 0:
            print(f"      嵌入 {min(s + EMB_BATCH, len(tensor))}/{len(tensor)}  "
                  f"{time.perf_counter() - t0:.0f}s")
    return emb


def l2norm(x: np.ndarray) -> np.ndarray:
    """
    按行 L2 归一化，使内积等价于余弦相似度。

    @author kk
    """
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_scores(emb: np.ndarray, pet_types: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造正负样本相似度数组。

    正样本 = 行内 front↔overall；负样本 = 跨行同姿态 max(FF, OO)，限同 pet_type。

    @author kk
    """
    front, overall = emb[0::2], emb[1::2]
    pos = np.einsum("ij,ij->i", front, overall)

    sim = np.maximum(front @ front.T, overall @ overall.T)
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    same_type = pet_types[iu[0]] == pet_types[iu[1]]
    return pos, sim[iu][same_type]


def roc_metrics(pos: np.ndarray, neg: np.ndarray) -> Dict[str, float]:
    """
    计算 ROC-AUC、EER 及低误报率下的召回，全部基于排序统计，不依赖 sklearn。

    @author kk
    """
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    order = scores.argsort()[::-1]
    labels = labels[order]

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / max(len(pos), 1)
    fpr = fp / max(len(neg), 1)

    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    eer_idx = int(np.argmin(np.abs((1 - tpr) - fpr)))
    out = {
        "auc": auc,
        "eer": float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2),
        "eer_threshold": float(scores[order][eer_idx]),
    }
    for target in (1e-2, 1e-3):
        idx = np.searchsorted(fpr, target, side="right") - 1
        out[f"tpr@fpr{target:g}"] = float(tpr[idx]) if idx >= 0 else 0.0
        out[f"thr@fpr{target:g}"] = float(scores[order][idx]) if idx >= 0 else 1.0
    return out


def report(tag: str, pos: np.ndarray, neg: np.ndarray, pet_types: np.ndarray) -> Dict[str, float]:
    """
    打印单一配置下的分布与指标，并返回指标字典。

    @author kk
    """
    m = roc_metrics(pos, neg)
    print(f"\n--- {tag} ---")
    print(f"正样本 {len(pos)} 对   均值 {pos.mean():.4f}  P10 {np.percentile(pos,10):.4f}  "
          f"P50 {np.percentile(pos,50):.4f}")
    print(f"负样本 {len(neg)} 对   均值 {neg.mean():.4f}  P90 {np.percentile(neg,90):.4f}  "
          f"P99 {np.percentile(neg,99):.4f}")
    print(f"AUC {m['auc']:.4f}   EER {m['eer']*100:.2f}%（阈值 {m['eer_threshold']:.4f}）")
    print(f"TPR@FPR=1%   {m['tpr@fpr0.01']*100:.2f}%（阈值 {m['thr@fpr0.01']:.4f}）")
    print(f"TPR@FPR=0.1% {m['tpr@fpr0.001']*100:.2f}%（阈值 {m['thr@fpr0.001']:.4f}）")
    for t in ("cat", "dog"):
        sub = pos[pet_types == t]
        print(f"  {t:<4} 正样本均值 {sub.mean():.4f}  P10 {np.percentile(sub,10):.4f}")
    return m


def main() -> None:
    """
    入口：读清单 → 检测 → 裁剪重抽嵌入 → 与整图基线同口径对比。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"清单记录 {len(manifest)} 行")

    # 本地图片路径以下载阶段落盘的 manifest 为准，不重新推断扩展名
    # （缓存中存在 .heic 等非常见后缀，按固定候选列表推断会漏）
    cache_manifest = pd.read_csv(os.path.join(CACHE_DIR, "manifest.csv"))
    path_of: Dict[Tuple[int, str], str] = {}
    for _, r in cache_manifest.iterrows():
        path_of[(int(r["row"]), "front")] = r["front_path"]
        path_of[(int(r["row"]), "overall")] = r["overall_path"]

    paths = [path_of[(int(r), k)] for r in manifest["row"] for k in ("front", "overall")]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 张缓存图片，首个：{missing[0]}")

    if os.path.exists(BOX_CACHE):
        boxes = pd.read_csv(BOX_CACHE)
        if len(boxes) == len(paths):
            print("[1/3] 命中检测缓存")
        else:
            boxes = None
    else:
        boxes = None
    if boxes is None:
        print(f"[1/3] YOLOv8n 检测 {len(paths)} 张")
        boxes = detect_all(paths)
        boxes.to_csv(BOX_CACHE, index=False, encoding="utf-8-sig")

    hit = boxes["cls"] >= 0
    print(f"      检出宠物 {int(hit.sum())}/{len(boxes)} ({hit.mean()*100:.1f}%)")
    print(f"      类别分布 cat={int((boxes['cls']==COCO_CAT).sum())} "
          f"dog={int((boxes['cls']==COCO_DOG).sum())} 未检出={int((~hit).sum())}")
    if hit.any():
        area_ratio = []
        for _, b in boxes[hit].iterrows():
            with Image.open(b["path"]) as im:
                w, h = im.size
            area_ratio.append((b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) / (w * h))
        area_ratio = np.array(area_ratio)
        print(f"      框面积占比 中位 {np.median(area_ratio)*100:.1f}%  "
              f"P10 {np.percentile(area_ratio,10)*100:.1f}%  P90 {np.percentile(area_ratio,90)*100:.1f}%")

    if os.path.exists(EMB_CROP_PATH) and np.load(EMB_CROP_PATH).shape[0] == len(paths):
        emb_crop = np.load(EMB_CROP_PATH)
        print("[2/3] 命中裁剪嵌入缓存")
    else:
        print(f"[2/3] 裁剪并抽取嵌入 {len(paths)} 张")
        tensor = np.zeros((len(paths), 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        for i, (p, (_, b)) in enumerate(zip(paths, boxes.iterrows())):
            box = (b["x1"], b["y1"], b["x2"], b["y2"]) if b["cls"] >= 0 else None
            arr = crop_and_preprocess(p, box)
            if arr is not None:
                tensor[i] = arr
        emb_crop = embed_batch(tensor)
        np.save(EMB_CROP_PATH, emb_crop)

    pet_types = manifest["pet_type"].to_numpy()
    print("\n[3/3] 同口径对比")

    emb_c = l2norm(emb_crop)
    pos_c, neg_c = build_scores(emb_c, pet_types)
    m_crop = report("检测裁剪（Stage 0）", pos_c, neg_c, pet_types)

    # 整图基线的嵌入按旧清单顺序存储，需按行号重建索引后对齐到当前清单
    old_manifest = pd.read_csv(os.path.join(CACHE_DIR, "manifest.csv"))
    old_manifest = old_manifest[old_manifest.front_ok & old_manifest.overall_ok].reset_index(drop=True)
    pos_in_old = {int(r): i for i, r in enumerate(old_manifest["row"])}
    emb_full_raw = np.load(EMB_FULL_PATH)
    idx = []
    for r in manifest["row"]:
        i = pos_in_old[int(r)]
        idx.extend([2 * i, 2 * i + 1])
    emb_f = l2norm(emb_full_raw[idx])
    pos_f, neg_f = build_scores(emb_f, pet_types)
    m_full = report("整图基线（对照）", pos_f, neg_f, pet_types)

    print("\n===== 结论 =====")
    print(f"{'指标':<16}{'整图':>10}{'裁剪':>10}{'变化':>10}")
    for key, label in [("auc", "AUC"), ("eer", "EER"),
                       ("tpr@fpr0.01", "TPR@FPR=1%"), ("tpr@fpr0.001", "TPR@FPR=0.1%")]:
        d = m_crop[key] - m_full[key]
        print(f"{label:<16}{m_full[key]:>10.4f}{m_crop[key]:>10.4f}{d:>+10.4f}")


if __name__ == "__main__":
    main()

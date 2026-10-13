"""
一致性识别基础数据集 —— 跨行重复注册扫描。

数据集每行是一次宠物注册（front 正脸图 + overall 全身图），行内两图必定属于同一只
宠物，因此可作为"同体相似度"的正样本参照，用来推导判定阈值。但同一个物理个体可能被
注册两次而占据两行，这些副本会污染 ReID 的负样本池（跨行对里混入真同体对），
需要先摸清规模。

流程：
  1. 下载并缓存全部图片（并发 + 重试 + 断点续传）
  2. DINOv2-small ONNX 批量抽取 CLS 嵌入
  3. 行内相似度分布 → 推导同体阈值
  4. 跨行行对相似度（同 pet_type，4 组组合取均值）→ 并查集聚类 → 报告重复簇
  5. 标签置换对照：打散 pet_type 后重跑聚类，作为偶然误聚的基线

@author kk
"""
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image

# 脚本所在目录即仓库根目录；全部路径以它为基准，避免硬编码个人机器的绝对路径导致换机不可跑
SCRIPT_DIR = Path(__file__).resolve().parent

# 源数据表与数据根目录：均可用环境变量覆盖，缺省落在仓库根目录
XLSX_PATH = str(Path(os.environ.get("PETREID_XLSX", SCRIPT_DIR / "一致性识别基础数据集.xlsx")))
PET_REID_DIR = str(Path(os.environ.get("PETREID_HOME", SCRIPT_DIR)))
CACHE_DIR = os.path.join(PET_REID_DIR, "_consistency_cache")
IMG_DIR = os.path.join(CACHE_DIR, "images")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.csv")
EMB_PATH = os.path.join(CACHE_DIR, "small_cls.npy")
REPORT_PATH = os.path.join(CACHE_DIR, "duplicate_scan.txt")

# ONNX 体积大未入库，缺省在数据根目录查找
ONNX_SMALL = str(Path(os.environ.get("PETREID_DINO_ONNX", Path(PET_REID_DIR) / "small_fp32.onnx")))

# 预处理参数，需与 dinov2-small 的 BitImageProcessor 保持一致
SHORT_EDGE = 256
CROP_SIZE = 224
PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DL_WORKERS = 24
DL_RETRY = 3
DL_TIMEOUT = 60
EMB_BATCH = 16

# 阈值百分位：取行内相似度分布的低分位，容忍少量失败配对
THRESHOLD_PCTS = (0.5, 1.0, 2.0, 5.0)
# 上传时间聚簇间隔，用于识别"同一会话连续注册"
SESSION_GAP_SEC = 180

URL_TS_RE = re.compile(r"/pet/(\d{14})")
EXT_RE = re.compile(r"(\.[A-Za-z0-9]{2,5})$")


@dataclass(frozen=True)
class Record:
    """一条注册记录（数据集的一行）。"""

    row: int
    pet_type: str
    front_path: str
    overall_path: str
    front_ts: Optional[pd.Timestamp]
    overall_ts: Optional[pd.Timestamp]


def parse_ts(url: str) -> Optional[pd.Timestamp]:
    """
    从 OSS 对象名中解析上传时间戳（形如 /pet/20260820233819...）。

    @author kk
    """
    if not isinstance(url, str):
        return None
    m = URL_TS_RE.search(url)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S", errors="coerce")


def local_name(row: int, kind: str, url: str) -> str:
    """
    生成本地缓存文件名；行号 + 图种已能保证唯一，扩展名沿用原始后缀。

    @author kk
    """
    m = EXT_RE.search(url.split("?")[0])
    ext = m.group(1).lower() if m else ".jpg"
    return f"{row:04d}_{kind}{ext}"


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗原始表并落成本地缓存清单：丢弃空行、解析时间戳、生成本地路径。

    @author kk
    """
    clean = df.dropna(subset=["front_image_url", "overall_image_url"]).reset_index(drop=True)
    records = []
    for i, r in clean.iterrows():
        f_url, o_url = str(r["front_image_url"]), str(r["overall_image_url"])
        records.append(
            {
                "row": i,
                "pet_type": str(r["pet_type"]),
                "front_url": f_url,
                "overall_url": o_url,
                "front_path": os.path.join(IMG_DIR, local_name(i, "front", f_url)),
                "overall_path": os.path.join(IMG_DIR, local_name(i, "overall", o_url)),
                "front_ts": parse_ts(f_url),
                "overall_ts": parse_ts(o_url),
            }
        )
    return pd.DataFrame.from_records(records)


def download_one(session: requests.Session, url: str, dest: str) -> Tuple[bool, str, int]:
    """
    下载单张图片，已存在则跳过；失败按 DL_RETRY 次重试。

    返回 (是否成功, 说明, 字节数)。异常在函数内消化，不向上抛，避免拖垮并发池。

    @author kk
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True, "cached", os.path.getsize(dest)
    last_err = ""
    for attempt in range(DL_RETRY):
        try:
            resp = session.get(url, timeout=DL_TIMEOUT)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                time.sleep(1.5 * (attempt + 1))
                continue
            # 校验确为可解码图像，避免把错误页落盘
            with Image.open(__import__("io").BytesIO(resp.content)) as im:
                im.verify()
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(resp.content)
            os.replace(tmp, dest)
            return True, "ok", len(resp.content)
        except Exception as exc:  # noqa: BLE001 - 网络异常种类多，统一重试
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return False, last_err, 0


def download_all(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    并发下载全部图片，并把下载结果回写到清单（新增 download_ok 列）。

    @author kk
    """
    os.makedirs(IMG_DIR, exist_ok=True)
    tasks = [(r["front_url"], r["front_path"]) for _, r in manifest.iterrows()]
    tasks += [(r["overall_url"], r["overall_path"]) for _, r in manifest.iterrows()]
    total = len(tasks)
    print(f"[1/4] 下载 {total} 张图片 -> {IMG_DIR}")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=DL_WORKERS, pool_maxsize=DL_WORKERS)
    session.mount("https://", adapter)
    ok: Dict[str, bool] = {}
    done = 0
    bytes_total = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = {pool.submit(download_one, session, u, d): d for u, d in tasks}
        for fut in as_completed(futures):
            dest = futures[fut]
            success, _msg, size = fut.result()
            ok[dest] = success
            done += 1
            bytes_total += size
            if done % 200 == 0 or done == total:
                rate = done / max(time.perf_counter() - t0, 1e-6)
                print(f"      {done}/{total}  成功 {sum(ok.values())}  已下 {bytes_total/1e6:.0f}MB  {rate:.1f} 张/秒")

    manifest = manifest.copy()
    manifest["front_ok"] = manifest["front_path"].map(ok).fillna(False).astype(bool)
    manifest["overall_ok"] = manifest["overall_path"].map(ok).fillna(False).astype(bool)
    bad = manifest[~(manifest["front_ok"] & manifest["overall_ok"])]
    print(f"      下载完成：可用记录 {len(manifest) - len(bad)}/{len(manifest)}")
    if len(bad):
        print(f"      失败记录行号：{list(bad['row'])[:20]}")
    return manifest


def preprocess(path: str) -> Optional[np.ndarray]:
    """
    按 BitImageProcessor 口径预处理：短边缩放到 256（双三次）、中心裁剪 224、归一化。

    返回 CHW float32 数组；解码失败（如 DNG 原始格式）返回 None。

    @author kk
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = SHORT_EDGE / min(w, h)
            new_w, new_h = max(SHORT_EDGE, int(round(w * scale))), max(SHORT_EDGE, int(round(h * scale)))
            im = im.resize((new_w, new_h), Image.BICUBIC)
            left = (new_w - CROP_SIZE) // 2
            top = (new_h - CROP_SIZE) // 2
            im = im.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001 - 解码异常一律视为不可用样本
        return None
    arr = (arr - PIXEL_MEAN) / PIXEL_STD
    return np.transpose(arr, (2, 0, 1))


def load_preprocessed(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量解码全部图片，返回 (张量 NCHW, 可用掩码)。

    @author kk
    """
    out = np.zeros((len(paths), 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    mask = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        arr = preprocess(p) if os.path.exists(p) else None
        if arr is not None:
            out[i] = arr
            mask[i] = True
    return out, mask


def embed(tensor: np.ndarray, mask: np.ndarray, onnx_path: str) -> np.ndarray:
    """
    用 ONNX Runtime 抽取 CLS 嵌入（选取第二个输出 select），并对不可用样本置零。

    @author kk
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    opts.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    # select 是 CLS 池化向量，embedding 是含 patch token 的完整序列
    sel_idx = out_names.index("select") if "select" in out_names else 1

    idx = np.flatnonzero(mask)
    dim = sess.get_outputs()[sel_idx].shape[-1]
    emb = np.zeros((len(tensor), dim), dtype=np.float32)
    print(f"[2/4] 抽取嵌入：{len(idx)} 张，维度 {dim}，batch={EMB_BATCH}")
    t0 = time.perf_counter()
    for s in range(0, len(idx), EMB_BATCH):
        batch_idx = idx[s : s + EMB_BATCH]
        outs = sess.run(None, {input_name: tensor[batch_idx]})
        emb[batch_idx] = outs[sel_idx]
        if (s // EMB_BATCH) % 20 == 0:
            print(f"      {min(s + EMB_BATCH, len(idx))}/{len(idx)}  {time.perf_counter() - t0:.0f}s")
    print(f"      嵌入完成，耗时 {time.perf_counter() - t0:.1f}s")
    return emb


def l2norm(x: np.ndarray) -> np.ndarray:
    """
    按行做 L2 归一化，使内积等价于余弦相似度。

    @author kk
    """
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class UnionFind:
    """
    并查集，用于把高于阈值的行对聚成连通簇。

    @author kk
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """路径压缩查找根节点。"""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """合并两个集合。"""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> Dict[int, List[int]]:
        """返回根节点 -> 成员列表的映射。"""
        groups: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            groups.setdefault(self.find(i), []).append(i)
        return groups


def row_pair_similarity(emb: np.ndarray) -> np.ndarray:
    """
    计算行对相似度矩阵：行内 4 种图组合（F-F, F-O, O-F, O-O）余弦相似度的均值。

    单组合易受某张图质量问题拖累，取 4 组合均值更稳健。

    @author kk
    """
    front = emb[0::2]
    overall = emb[1::2]
    ff = front @ front.T
    oo = overall @ overall.T
    fo = front @ overall.T
    return (ff + oo + fo + fo.T) / 4.0


def cluster_rows(
    sim: np.ndarray, pet_types: np.ndarray, threshold: float
) -> Tuple[UnionFind, np.ndarray]:
    """
    在同行宠物类型且相似度 >= 阈值的行对之间做并查集合并。

    返回 (并查集, 触发合并的行对索引数组)。

    @author kk
    """
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_sim = sim[iu]
    same_type = pet_types[iu[0]] == pet_types[iu[1]]
    hit = (pair_sim >= threshold) & same_type
    uf = UnionFind(n)
    for a, b in zip(iu[0][hit], iu[1][hit]):
        uf.union(int(a), int(b))
    return uf, np.column_stack([iu[0][hit], iu[1][hit]])


def cluster_stats(uf: UnionFind, n: int) -> Tuple[int, int, Dict[int, int]]:
    """
    统计聚类结果：涉及行数、簇数、簇大小分布。

    @author kk
    """
    groups = uf.clusters()
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    involved = sum(len(v) for v in multi.values())
    sizes: Dict[int, int] = {}
    for v in multi.values():
        sizes[len(v)] = sizes.get(len(v), 0) + 1
    return involved, len(multi), dict(sorted(sizes.items()))


def session_id(ts: pd.Timestamp, boundaries: List[pd.Timestamp]) -> int:
    """
    按全局时间轴上的间隔切分"上传会话"；间隔超过 SESSION_GAP_SEC 则开新会话。

    注意：数据集无用户标识，全局时间切分只是代理口径，不同用户的并发上传会被并到
    同一会话里，因此该字段仅作辅助线索，不作为个体判定依据。

    @author kk
    """
    if pd.isna(ts):
        return -1
    import bisect

    return bisect.bisect_right(boundaries, ts)


def main() -> None:
    """
    入口：下载 → 抽取嵌入 → 行内定阈值 → 跨行聚类 → 置换对照 → 输出报告。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - 老终端不支持时忽略
        pass
    os.makedirs(CACHE_DIR, exist_ok=True)

    df = pd.read_excel(XLSX_PATH)
    print(f"原始表：{len(df)} 行")
    manifest = build_manifest(df)
    print(f"清洗后：{len(manifest)} 行（丢弃 {len(df) - len(manifest)} 行空值）")

    if os.path.exists(MANIFEST_PATH) and all(
        os.path.exists(p) for p in list(manifest["front_path"])[:50] + list(manifest["overall_path"])[:50]
    ):
        cached = pd.read_csv(MANIFEST_PATH)
        if len(cached) == len(manifest):
            print("[1/4] 命中本地缓存，跳过下载")
            # CSV 往返会把时间戳降级成字符串，需还原成 Timestamp 供后续间隔计算
            for col in ("front_ts", "overall_ts"):
                cached[col] = pd.to_datetime(cached[col], errors="coerce")
            manifest = cached
    if "front_ok" not in manifest.columns:
        manifest = download_all(manifest)
        manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    usable = manifest[manifest["front_ok"] & manifest["overall_ok"]].reset_index(drop=True)
    print(f"      可用记录 {len(usable)}/{len(manifest)}")

    paths = list(usable["front_path"]) + list(usable["overall_path"])
    # 行主序交错：2*i = front, 2*i+1 = overall
    order = [p for pair in zip(usable["front_path"], usable["overall_path"]) for p in pair]
    tensor, mask = load_preprocessed(order)
    print(f"      可解码 {mask.sum()}/{len(order)} 张")

    if os.path.exists(EMB_PATH) and np.load(EMB_PATH).shape[0] == mask.sum():
        raw = np.load(EMB_PATH)
        print("[2/4] 命中嵌入缓存")
    else:
        raw = embed(tensor, mask, ONNX_SMALL)
        # 仅保留可解码样本的嵌入，并压缩成 (可用张数, D)
        raw = raw[mask]
        np.save(EMB_PATH, raw)

    # 仅保留两张图都可解码的记录，保证行内配对完整
    ok_arr = mask.reshape(-1, 2).all(axis=1)
    kept = int(ok_arr.sum())
    print(f"      行内配对完整记录 {kept}/{len(usable)}")
    sel = np.repeat(ok_arr, 2)
    emb = l2norm(raw[sel])
    pet_types = np.array(usable.loc[ok_arr, "pet_type"].tolist())
    rows = np.array(usable.loc[ok_arr, "row"].tolist())

    front = emb[0::2]
    overall = emb[1::2]
    intra = np.einsum("ij,ij->i", front, overall)

    print("\n[3/4] 行内相似度分布（同体正样本参照）")
    qs = [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99]
    for q in qs:
        print(f"      P{q:<5} {np.percentile(intra, q):.4f}")
    print(f"      均值 {intra.mean():.4f}  标准差 {intra.std():.4f}")
    for t in ("cat", "dog"):
        sub = intra[pet_types == t]
        print(f"      {t:<4} n={len(sub):<5} 均值 {sub.mean():.4f}  P1 {np.percentile(sub, 1):.4f}")

    # 行内相似度逼近 1 说明 front/overall 实为同一张图，这类记录不能当"跨姿态正样本"
    near_one = intra >= 0.999
    print(f"\n      行内相似度 >=0.999 的记录：{near_one.sum()} / {len(intra)} "
          f"({near_one.mean()*100:.1f}%)")
    if near_one.any():
        import hashlib

        same_bytes = 0
        checked = np.flatnonzero(near_one)
        for i in checked:
            fp = usable.loc[ok_arr, "front_path"].iloc[i]
            op = usable.loc[ok_arr, "overall_path"].iloc[i]
            try:
                with open(fp, "rb") as fa, open(op, "rb") as fb:
                    if hashlib.md5(fa.read()).digest() == hashlib.md5(fb.read()).digest():
                        same_bytes += 1
            except OSError:
                continue
        print(f"      其中字节级完全相同（同一张图被存两份）：{same_bytes}")
        print(f"      示例行号：{[int(rows[i]) for i in checked[:15]]}")

    sim = row_pair_similarity(emb)
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    cross = sim[iu]
    same = pet_types[iu[0]] == pet_types[iu[1]]
    print("\n      跨行行对相似度（负样本参照）")
    for q in (50, 90, 99, 99.9, 100):
        print(f"      P{q:<5} {np.percentile(cross[same], q):.4f}")

    # 每行在"其他行"中的最近邻相似度，用于观察是否成对出现
    same_row = pet_types[:, None] == pet_types[None, :]
    masked = np.where(same_row, sim, -1.0)
    np.fill_diagonal(masked, -1.0)
    top1 = masked.max(axis=1)
    print(f"      跨行 Top-1 相似度：均值 {top1.mean():.4f}  P50 {np.percentile(top1,50):.4f}  "
          f"P90 {np.percentile(top1,90):.4f}  P99 {np.percentile(top1,99):.4f}")

    print("\n[4/4] 阈值聚类")
    all_ts = sorted([t for t in list(usable["front_ts"]) + list(usable["overall_ts"]) if pd.notna(t)])
    boundaries = [all_ts[0]]
    for prev, cur in zip(all_ts, all_ts[1:]):
        if (cur - prev).total_seconds() > SESSION_GAP_SEC:
            boundaries.append(cur)
    boundaries = boundaries[1:]

    rng = np.random.default_rng(42)
    report_lines = []
    for pct in THRESHOLD_PCTS:
        thr = float(np.percentile(intra, pct))
        uf, pairs = cluster_rows(sim, pet_types, thr)
        involved, nclust, sizes = cluster_stats(uf, n)

        # 置换对照：打散宠物类型标签，衡量偶然误聚的规模
        shuffled = pet_types[rng.permutation(n)]
        uf_null, _ = cluster_rows(sim, shuffled, thr)
        inv_null, nclust_null, _ = cluster_stats(uf_null, n)

        print(f"\n      阈值 = 行内 P{pct} = {thr:.4f}")
        print(f"        高于阈值行对 {len(pairs)}   涉及记录 {involved}   簇数 {nclust}")
        print(f"        簇大小分布 {sizes}")
        print(f"        置换对照：涉及记录 {inv_null}   簇数 {nclust_null}")
        report_lines.append((pct, thr, len(pairs), involved, nclust, sizes, inv_null, nclust_null))

        if pct == 1.0 and len(pairs):
            gap_secs = []
            for a, b in pairs:
                ta = usable.loc[ok_arr, "front_ts"].iloc[a]
                tb = usable.loc[ok_arr, "front_ts"].iloc[b]
                if pd.notna(ta) and pd.notna(tb):
                    gap_secs.append(abs((tb - ta).total_seconds()))
            gap_secs = np.array(gap_secs)
            if len(gap_secs):
                print(f"        疑似对上传间隔中位数 {np.median(gap_secs):.0f}s  "
                      f"<SESSION_GAP 占比 {(gap_secs < SESSION_GAP_SEC).mean()*100:.1f}%")
            groups = uf.clusters()
            multi = sorted([v for v in groups.values() if len(v) > 1], key=len, reverse=True)[:10]
            print("        最大簇（行号 / 类型 / 两两相似度）：")
            for g in multi[:5]:
                rws = [int(rows[i]) for i in g]
                types = {pet_types[i] for i in g}
                ps = [float(sim[a][b]) for i, a in enumerate(g) for b in g[i + 1 :]]
                print(f"          {rws} {types} 相似度 {[round(x, 3) for x in ps]}")

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("阈值百分位\t阈值\t高于阈值行对\t涉及记录\t簇数\t簇大小分布\t置换-涉及记录\t置换-簇数\n")
        for r in report_lines:
            fh.write(f"{r[0]}\t{r[1]:.4f}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}\t{r[7]}\n")
    print(f"\n报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()

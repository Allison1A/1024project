"""
一致性识别基础数据集 —— 清洗与 train/test 划分清单生成。

数据集每行是一次宠物注册（front 正脸图 + overall 全身图），行内两图属于同一只宠物。
前置扫描（scan_consistency_dups.py）已查明三类需要处理的质量问题，其结论以常量形式
固化在本文件中，不在此重新计算：

  1. 空值行与下载失败行 —— 无可用图片，直接剔除
  2. front 与 overall 为字节级同一张图 —— 正样本退化为"图与自身"，学不到跨姿态
     不变性，直接剔除
  3. 重复注册簇 —— 同一只物理宠物被注册成多行，这些行必须共享同一个 iid，
     否则跨行负样本池里会混入真同体对，且 train/test 划分会造成泄漏

行号口径（全文件统一）：读取 xlsx 后执行
    dropna(subset=["front_image_url", "overall_image_url"]).reset_index(drop=True)
所得 DataFrame 的 0-based 索引即为"行号"。注意空值行 79 在 dropna 后已不在索引中，
故 DROP_EMPTY_ROWS 仅作来源记录，不参与实际过滤。

@author kk
"""
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# 脚本所在目录，用于缺省定位同目录下的源数据表与输出清单
SCRIPT_DIR = Path(__file__).resolve().parent

# 源数据表：优先读环境变量 PETREID_XLSX，缺省回落到脚本同目录下的默认文件名
XLSX_PATH = str(Path(os.environ.get("PETREID_XLSX", SCRIPT_DIR / "一致性识别基础数据集.xlsx")))

# 输出清单：优先读环境变量 PETREID_MANIFEST_OUT，缺省落到脚本同目录下的 clean_manifest.csv
OUTPUT_PATH = str(Path(os.environ.get("PETREID_MANIFEST_OUT", SCRIPT_DIR / "clean_manifest.csv")))

URL_COLUMNS = ("front_image_url", "overall_image_url")
OUTPUT_COLUMNS = ("iid", "row", "pet_type", "split", "front_image_url", "overall_image_url")

# 原始表中两个 URL 均为空的行；dropna 后即消失，此处仅作来源记录
DROP_EMPTY_ROWS: Tuple[int, ...] = (79,)

# OSS 取图失败（重试 3 次仍失败）的行
DOWNLOAD_FAILED_ROWS: Tuple[int, ...] = (354,)

# front 与 overall 为字节级相同的同一张图（MD5 全量比对得出，63 行）
SELF_DUP_ROWS: Tuple[int, ...] = (
    5, 30, 39, 57, 63, 64, 73, 80, 96, 102, 111, 142, 158, 172, 178, 194,
    266, 298, 299, 304, 316, 319, 321, 385, 401, 402, 498, 503, 509, 523,
    542, 551, 554, 576, 610, 618, 632, 675, 700, 727, 730, 737, 747, 749,
    750, 771, 791, 816, 820, 836, 852, 867, 894, 898, 905, 911, 924, 972,
    994, 1011, 1023, 1049, 1064,
)

# 重复注册簇：簇内各行为同一只物理宠物。
# 由 MD5 字节级重复与 DINOv2 嵌入相似度（同姿态 max(FF,OO) >= 0.90）两路证据
# 取并集后做传递合并得出，共 21 簇。
DUPLICATE_CLUSTERS: Tuple[Tuple[int, ...], ...] = (
    (99, 101), (147, 151), (172, 178), (192, 420), (201, 422), (231, 232),
    (240, 936), (320, 704, 705), (322, 785), (324, 486), (414, 415),
    (470, 973), (487, 779), (556, 582), (617, 630), (676, 715), (682, 684),
    (725, 728), (834, 840), (989, 990), (1014, 1016),
)

TEST_RATIO = 0.2
RANDOM_SEED = 42

# 自检基线，来自前置扫描
EXPECTED_RECORDS = 1002
EXPECTED_IIDS = 981
EXPECTED_ACTIVE_CLUSTERS = 20


def load_indexed(xlsx_path: str) -> pd.DataFrame:
    """
    读取原始 xlsx 并建立统一行号口径。

    丢弃两个 URL 任一为空的行后重排索引，并把该索引固化为 row 列，使后续所有
    步骤都以同一套行号对话。

    @author kk
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"数据集不存在：{xlsx_path}")
    raw = pd.read_excel(xlsx_path)
    missing = [c for c in ("pet_type", *URL_COLUMNS) if c not in raw.columns]
    if missing:
        raise ValueError(f"数据集缺少必要列：{missing}")
    indexed = raw.dropna(subset=list(URL_COLUMNS)).reset_index(drop=True)
    return indexed.assign(row=indexed.index.to_numpy())


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """
    剔除不可用记录：下载失败行与 front==overall 的自重复行。

    @author kk
    """
    banned: Set[int] = set(DOWNLOAD_FAILED_ROWS) | set(SELF_DUP_ROWS)
    return df[~df["row"].isin(banned)].reset_index(drop=True)


def active_clusters(surviving: Set[int]) -> Tuple[Tuple[int, ...], ...]:
    """
    把重复注册簇裁剪到存活行，并丢弃裁剪后不足 2 行的簇。

    例如簇 (172, 178) 的两行都在 SELF_DUP_ROWS 中被剔除，该簇整体失效。

    @author kk
    """
    trimmed = []
    for cluster in DUPLICATE_CLUSTERS:
        kept = tuple(r for r in cluster if r in surviving)
        if len(kept) > 1:
            trimmed.append(kept)
    return tuple(trimmed)


def build_iid_map(df: pd.DataFrame) -> Tuple[Dict[int, int], Tuple[Tuple[int, ...], ...]]:
    """
    为每条存活记录分配 iid（物理个体标识）。

    同一重复簇内的存活行共享该簇最小行号作为原始键，其余行以自身行号为键；
    最后把原始键重新编号为从 0 开始的连续整数。

    返回 (行号 -> iid 映射, 生效的重复簇)。

    @author kk
    """
    surviving: Set[int] = set(df["row"].tolist())
    clusters = active_clusters(surviving)

    raw_key: Dict[int, int] = {r: r for r in surviving}
    for cluster in clusters:
        anchor = min(cluster)
        for r in cluster:
            raw_key[r] = anchor

    ordered_keys = sorted(set(raw_key.values()))
    renumber = {k: i for i, k in enumerate(ordered_keys)}
    return {r: renumber[k] for r, k in raw_key.items()}, clusters


def assign_iid(df: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[Tuple[int, ...], ...]]:
    """
    把 iid 映射落成 DataFrame 的新列，返回新对象。

    @author kk
    """
    iid_map, clusters = build_iid_map(df)
    return df.assign(iid=df["row"].map(iid_map).astype(int)), clusters


def iid_pet_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总每个 iid 的宠物类型，供分层抽样使用。

    同一 iid 的各行理论上类型一致（簇聚类已限定同类），若出现分歧则取首个值并
    在摘要中可见，不静默改写数据。

    @author kk
    """
    return (
        df.groupby("iid", as_index=False)
        .agg(pet_type=("pet_type", "first"), n_rows=("row", "size"))
        .sort_values("iid")
        .reset_index(drop=True)
    )


def split_by_iid(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 iid 为单位做 train/test 划分，并在 pet_type 上分层。

    关键约束：同一 iid 的所有行必须落在同一侧。以 iid 而非行为抽样单位即可保证，
    避免同一只宠物的两条注册记录被拆到训练集与测试集两边造成泄漏。

    @author kk
    """
    groups = iid_pet_type(df)
    rng = np.random.default_rng(RANDOM_SEED)

    test_iids: Set[int] = set()
    for pet_type in sorted(groups["pet_type"].unique()):
        candidates = groups.loc[groups["pet_type"] == pet_type, "iid"].to_numpy()
        shuffled = rng.permutation(candidates)
        n_test = int(round(len(shuffled) * TEST_RATIO))
        test_iids.update(int(x) for x in shuffled[:n_test])

    split = np.where(df["iid"].isin(test_iids), "test", "train")
    return df.assign(split=split)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    整理输出列顺序并按 iid、row 排序。

    @author kk
    """
    return (
        df[list(OUTPUT_COLUMNS)]
        .sort_values(["iid", "row"])
        .reset_index(drop=True)
    )


def print_summary(df: pd.DataFrame, clusters: Tuple[Tuple[int, ...], ...]) -> None:
    """
    打印清洗与划分统计摘要，并对照前置扫描的预期基线自检。

    @author kk
    """
    n_records = len(df)
    n_iids = df["iid"].nunique()
    print("===== 清洗结果 =====")
    print(f"存活记录        {n_records}")
    print(f"独立 iid        {n_iids}")
    print(f"生效重复簇      {len(clusters)}（覆盖 {sum(len(c) for c in clusters)} 行）")
    print(f"剔除 自重复     {len(SELF_DUP_ROWS)}")
    print(f"剔除 下载失败   {len(DOWNLOAD_FAILED_ROWS)}")
    print(f"剔除 空值       {len(DROP_EMPTY_ROWS)}（dropna 阶段已移除）")

    print("\n===== 类型分布 =====")
    print(f"{'':8}{'记录':>8}{'iid':>8}")
    groups = iid_pet_type(df)
    for pet_type in sorted(df["pet_type"].unique()):
        n_r = int((df["pet_type"] == pet_type).sum())
        n_i = int((groups["pet_type"] == pet_type).sum())
        print(f"{pet_type:<8}{n_r:>8}{n_i:>8}")

    print("\n===== train/test 划分（以 iid 为单位）=====")
    print(f"{'split':<8}{'记录':>8}{'iid':>8}{'cat':>8}{'dog':>8}")
    for split in ("train", "test"):
        sub = df[df["split"] == split]
        sub_iids = sub["iid"].nunique()
        n_cat = int((sub["pet_type"] == "cat").sum())
        n_dog = int((sub["pet_type"] == "dog").sum())
        print(f"{split:<8}{len(sub):>8}{sub_iids:>8}{n_cat:>8}{n_dog:>8}")
    test_share = df["split"].eq("test").mean()
    print(f"test 记录占比   {test_share * 100:.1f}%")

    multi = df.groupby("iid").size()
    print(f"\n多行 iid（重复注册）{int((multi > 1).sum())} 个")
    for iid in multi[multi > 1].index[:10]:
        sub = df[df["iid"] == iid]
        print(f"  iid={iid:<5} 行号 {list(sub['row'])} 类型 {sub['pet_type'].iloc[0]} "
              f"split {sub['split'].iloc[0]}")

    leaked = df.groupby("iid")["split"].nunique()
    n_leaked = int((leaked > 1).sum())
    print(f"\n泄漏检查：跨 split 的 iid {n_leaked} 个" + ("（正常）" if n_leaked == 0 else "（异常！）"))

    mismatch = [
        ("存活记录", n_records, EXPECTED_RECORDS),
        ("独立 iid", n_iids, EXPECTED_IIDS),
        ("生效重复簇", len(clusters), EXPECTED_ACTIVE_CLUSTERS),
    ]
    bad = [(name, got, exp) for name, got, exp in mismatch if got != exp]
    if bad:
        for name, got, exp in bad:
            print(f"[自检不符] {name}: 实际 {got}，预期 {exp}")
    else:
        print("自检通过：记录数、iid 数、重复簇数均与前置扫描一致")


def main() -> None:
    """
    入口：读表 → 剔除不可用 → 分配 iid → 按 iid 划分 → 落盘 + 摘要。

    @author kk
    """
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - 老终端不支持时忽略
        pass

    indexed = load_indexed(XLSX_PATH)
    print(f"原始表（去空值后）{len(indexed)} 行")

    usable = drop_unusable(indexed)
    with_iid, clusters = assign_iid(usable)
    split = split_by_iid(with_iid)
    result = finalize(split)

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print_summary(result, clusters)
    print(f"\n清单已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

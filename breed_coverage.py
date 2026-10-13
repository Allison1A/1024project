"""
品种覆盖度盘点 —— 零样本品种打标。

动机：命题验收标准含「品种覆盖 ≥10 种」，但业务数据只提供 pet_type（cat/dog），
无品种字段。此前把「猫狗双物种」当成「品种覆盖」是口径错误：物种是 cat/dog 两类，
品种是金毛/柯基/英短/美短这一层。本脚本用零样本图文模型对已落盘的 2004 张图打品种标签，
给出「实际覆盖多少个品种」的可复现口径。

口径说明（重要，别混）：
  - **品种（breed）** = 金毛、柯基、英短、美短…… 命题要的是这一层
  - **花色（coat pattern）** = 橘猫、三花、玳瑁…… 中文养宠语境常当品种叫，但
    遗传上只是毛色表现，同一个品种可以有多花色。本脚本两者都打，但分开统计，
    报告中只把真正的 breed 计入「品种覆盖」
  - **物种（species）** = cat/dog，只是最粗的一层

零样本打标结果仅作**覆盖度估计**，不作为评测标签使用——最终品种标注应取业务方
已有的「品种分类标准」（见命题介绍「已有资源」栏）。

@author kk
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from stage0_detect_crop import CACHE_DIR, MANIFEST_PATH

MODEL_ID = "AI-ModelScope/chinese-clip-vit-base-patch16"
CACHE_PATH = os.path.join(CACHE_DIR, "breed_labels.npz")
# 每张图缩到这个边长送进 CLIP，够用且省时（2004 张 CPU 推理）
CLIP_SIZE = 224
BATCH = 32

# 真正的品种：key 为规范名，value 为送进模型的中文提示词
DOG_BREEDS: Dict[str, str] = {
    "金毛寻回犬": "一只金毛寻回犬",
    "威尔士柯基犬": "一只柯基犬",
    "拉布拉多": "一只拉布拉多犬",
    "贵宾/泰迪": "一只泰迪犬",
    "比熊": "一只比熊犬",
    "博美": "一只博美犬",
    "哈士奇": "一只哈士奇",
    "阿拉斯加": "一只阿拉斯加雪橇犬",
    "边境牧羊犬": "一只边境牧羊犬",
    "萨摩耶": "一只萨摩耶犬",
    "柴犬": "一只柴犬",
    "吉娃娃": "一只吉娃娃",
    "雪纳瑞": "一只雪纳瑞",
    "法国斗牛犬": "一只法国斗牛犬",
    "巴哥犬": "一只巴哥犬",
    "腊肠犬": "一只腊肠犬",
    "德国牧羊犬": "一只德国牧羊犬",
    "杜宾": "一只杜宾犬",
    "中华田园犬": "一只中华田园犬",
}

CAT_BREEDS: Dict[str, str] = {
    "英国短毛猫": "一只英国短毛猫",
    "美国短毛猫": "一只美国短毛猫",
    "布偶猫": "一只布偶猫",
    "异国短毛猫(加菲)": "一只加菲猫",
    "暹罗猫": "一只暹罗猫",
    "波斯猫": "一只波斯猫",
    "缅因猫": "一只缅因猫",
    "斯芬克斯无毛猫": "一只无毛猫",
    "孟加拉豹猫": "一只孟加拉豹猫",
    "中华田园猫": "一只中华田园猫",
}

# 花色：中文语境常被叫成「品种」，但属于毛色表现，单独统计、不计入品种覆盖
COAT_PATTERNS: Dict[str, str] = {
    "橘猫": "一只橘色的猫",
    "狸花": "一只狸花猫",
    "三花": "一只三花猫",
    "玳瑁": "一只玳瑁色的猫",
    "奶牛(黑白)": "一只黑白花色的猫",
    "银渐层": "一只银渐层猫",
    "金渐层": "一只金渐层猫",
    "蓝猫(纯灰)": "一只纯灰色的猫",
}


def resolve_paths(manifest: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    把清洗后清单展开为 front/overall 交错的本地路径与对应 pet_type。

    路径一律查下载阶段的 manifest（缓存中存在 .heic/.dng 等后缀，按扩展名推断会漏）。

    @author kk
    """
    cache = pd.read_csv(os.path.join(CACHE_DIR, "manifest.csv"))
    path_of: Dict[Tuple[int, str], str] = {}
    for _, r in cache.iterrows():
        path_of[(int(r["row"]), "front")] = r["front_path"]
        path_of[(int(r["row"]), "overall")] = r["overall_path"]

    paths: List[str] = []
    types: List[str] = []
    for r, t in zip(manifest["row"], manifest["pet_type"]):
        for k in ("front", "overall"):
            paths.append(path_of[(int(r), k)])
            types.append(t)
    return paths, np.asarray(types)


def load_batch(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    读入一批图片并缩放到 CLIP 输入尺寸，返回 float32 数组与成功掩码。

    读图失败者标为失败并在后续统计中剔除，不用零张量冒充。

    @author kk
    """
    arr = np.zeros((len(paths), CLIP_SIZE, CLIP_SIZE, 3), dtype=np.uint8)
    ok = np.ones(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((CLIP_SIZE, CLIP_SIZE), Image.BICUBIC)
                arr[i] = np.asarray(im, dtype=np.uint8)
        except (OSError, ValueError):
            ok[i] = False
    return arr, ok


def text_features(model, proc, texts: List[str]):
    """
    手工提取文本嵌入。

    不能直接用 model.get_text_features：本模型的 text_model 返回的 pooler_output 为 None，
    而 get_text_features 内部正是取 pooler_output 再过 text_projection，会抛
    「linear(): argument 'input' must be Tensor, not NoneType」。故显式取 [CLS] 位置
    （last_hidden_state[:, 0]）——这也是 ChineseCLIP 原始实现的对齐方式。

    @author kk
    """
    import torch

    with torch.no_grad():
        out = model.text_model(**proc(text=texts, return_tensors="pt", padding=True))
        feat = model.text_projection(out.last_hidden_state[:, 0])
    return feat / feat.norm(dim=-1, keepdim=True)


def image_features(model, proc, images):
    """
    手工提取图像嵌入，理由同 text_features（同样绕开 pooler_output 为 None 的问题）。

    @author kk
    """
    import torch

    pixel = proc(images=images, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        feat = model.get_image_features(pixel_values=pixel)
    return feat / feat.norm(dim=-1, keepdim=True)


def label_all(paths: List[str], pet_types: np.ndarray) -> Dict[str, np.ndarray]:
    """
    用零样本模型给所有图片打分并取最高分为预测标签，命中缓存则直接读盘。

    分两轮：第一轮在 19 个犬种 + 10 个猫种上判品种；第二轮在 8 种花色上判毛色。
    两轮共用同一次图像编码，避免重复推理。

    @author kk
    """
    if os.path.exists(CACHE_PATH):
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if len(cached["breed_idx"]) == len(paths):
            print(f"      命中缓存 {os.path.basename(CACHE_PATH)}")
            return {k: cached[k] for k in cached.files}

    from modelscope import snapshot_download
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    model_dir = snapshot_download(MODEL_ID)
    model = ChineseCLIPModel.from_pretrained(model_dir).eval()
    proc = ChineseCLIPProcessor.from_pretrained(model_dir)

    breeds = {**DOG_BREEDS, **CAT_BREEDS}
    breed_names = list(breeds)
    coat_names = list(COAT_PATTERNS)
    n_breed = len(breed_names)
    txt_feat = text_features(model, proc,
                             list(breeds.values()) + list(COAT_PATTERNS.values()))

    breed_idx = np.full(len(paths), -1, dtype=np.int32)
    coat_idx = np.full(len(paths), -1, dtype=np.int32)
    ok_all = np.zeros(len(paths), dtype=bool)

    for s in range(0, len(paths), BATCH):
        e = min(s + BATCH, len(paths))
        arr, ok = load_batch(paths[s:e])
        sim = (image_features(model, proc, list(arr)) @ txt_feat.T).numpy()
        breed_idx[s:e] = sim[:, :n_breed].argmax(axis=1)
        coat_idx[s:e] = sim[:, n_breed:].argmax(axis=1)
        ok_all[s:e] = ok
        if (s // BATCH) % 20 == 0:
            print(f"      打标 {e}/{len(paths)}")

    out = {
        "breed_idx": breed_idx,
        "coat_idx": coat_idx,
        "ok": ok_all,
        "breed_names": np.array(breed_names, dtype=object),
        "coat_names": np.array(coat_names, dtype=object),
        "pet_types": pet_types,
    }
    np.savez(CACHE_PATH, **out)
    return out


def consistency_check(res: Dict[str, np.ndarray]) -> None:
    """
    检验同一只宠物的 front 与 overall 两张图是否得到同一个品种标签。

    这是标签可靠性的直接度量：同一次注册的两张图必然是同一只宠物，若打标结果不一致，
    说明零样本模型在该品种上不稳定（毛色相似品种尤其容易互混，如泰迪/比熊同为卷毛小型犬）。
    一致率低于阈值时，本脚本的覆盖度数字只能当量级参考，不能当结论。

    @author kk
    """
    ok, b_idx = res["ok"], res["breed_idx"]
    names = list(res["breed_names"])
    pet_types = res["pet_types"]

    pair_ok = ok[0::2] & ok[1::2]
    f, o = b_idx[0::2], b_idx[1::2]
    agree = pair_ok & (f == o)

    print("\n===== 标签自洽性（同一注册的 front/overall 应同品种）=====")
    for sp, label in (("cat", "猫"), ("dog", "狗")):
        m = pair_ok & (pet_types[0::2] == sp)
        if m.sum() == 0:
            continue
        a = float((agree & (pet_types[0::2] == sp)).sum() / m.sum())
        print(f"  {label}：{a * 100:.2f}%（{int(m.sum())} 对）")

    top = pd.Series([names[i] for i in f[pair_ok]]).value_counts().head(12)
    print("\n  最常被两张图判为同品种的品种（即打标最稳定的）：")
    for name, c in top.items():
        sel = pair_ok & (np.array([names[i] for i in f]) == name)
        r = float((agree & sel).sum() / max(sel.sum(), 1))
        print(f"    {name:<18}{r * 100:>7.2f}%  （{int(sel.sum())} 对）")


def report(res: Dict[str, np.ndarray]) -> None:
    """
    打印品种与花色的覆盖统计，并指出「达到 10 个品种」这一验收线的实测差距。

    @author kk
    """
    ok = res["ok"]
    breed_names = list(res["breed_names"])
    coat_names = list(res["coat_names"])
    b_idx, c_idx = res["breed_idx"], res["coat_idx"]

    print(f"\n打标成功 {int(ok.sum())}/{len(ok)} 张")
    valid = len(b_idx)

    for sp, label in (("dog", "狗"), ("cat", "猫")):
        sel = ok & (b_idx >= 0)
        # 按物种分别统计：狗图只应命中犬种，若大量命中猫种说明零样本模型不可靠
        species_of_breed = np.array([0 if n in DOG_BREEDS else 1 for n in breed_names])
        want = 0 if sp == "dog" else 1
        correct = sel & (species_of_breed[b_idx] == want)
        n_img = int(correct.sum())
        print(f"\n--- {label}（打标 {n_img} 张，按物种过滤后）---")
        if n_img == 0:
            print("  无有效样本")
            continue
        counts = pd.Series([breed_names[i] for i in b_idx[correct]]).value_counts()
        cover = int((counts >= 5).sum())
        print(f"  命中品种数 {len(counts)}，其中样本 ≥5 张的 {cover} 个")
        for name, c in counts.items():
            print(f"    {name:<18}{c:>6} 张{c / n_img * 100:>7.2f}%")

        coats = pd.Series([coat_names[i] for i in c_idx[correct]]).value_counts()
        print(f"  花色分布（不计入品种覆盖）：")
        for name, c in coats.head(5).items():
            print(f"    {name:<18}{c:>6} 张")

    print("\n===== 品种覆盖（口径：样本 ≥5 张才计为一个有效品种）=====")
    species_of_breed = np.array([0 if n in DOG_BREEDS else 1 for n in breed_names])
    total = 0
    for want, label in ((0, "狗"), (1, "猫")):
        sel = ok & (species_of_breed[b_idx] == want)
        counts = pd.Series([breed_names[i] for i in b_idx[sel]]).value_counts()
        cover = counts[counts >= 5]
        total += len(cover)
        print(f"  {label}：{len(cover)} 个品种（{', '.join(cover.index[:12])}）")
    print(f"  合计：{total} 个品种   验收线 ≥10 种 → "
          f"{'达标' if total >= 10 else '未达标'}")
    print("\n注：零样本打标仅作覆盖度估计，非评测标签。"
          "正式品种标注应取业务方已有的品种分类标准。")


def main() -> None:
    """
    入口：解析路径 → 零样本打标 → 报告品种覆盖。

    @author kk
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    manifest = pd.read_csv(MANIFEST_PATH)
    paths, pet_types = resolve_paths(manifest)
    print(f"清单 {len(manifest)} 行 / 图片 {len(paths)} 张"
          f"（猫 {int((pet_types == 'cat').sum())} / 狗 {int((pet_types == 'dog').sum())}）")
    print("口径：物种 = cat/dog 两层；品种 = 金毛/柯基/英短/美短；花色 = 橘猫/三花/狸花")

    res = label_all(paths, pet_types)
    report(res)
    consistency_check(res)


if __name__ == "__main__":
    main()

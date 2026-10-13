"""
推理延迟基准：PyTorch vs ONNX FP32 vs ONNX INT8，base vs small 模型。

用于回填报名文案中"双图推理含检测"的实测数字。
测量口径：预热 2 次后取 5 次平均，单图 batch=1，双图 batch=2。

@author kk
"""
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# 脚本所在目录，用于定位同目录下的 ONNX 模型文件
SCRIPT_DIR = Path(__file__).resolve().parent

# 模型缓存根目录：优先读环境变量 PETREID_MODEL_CACHE，缺省回落到用户主目录下 modelscope 的 facebook 子目录
_MODEL_CACHE_ROOT = Path(
    os.environ.get(
        "PETREID_MODEL_CACHE",
        Path.home() / ".cache" / "modelscope" / "hub" / "models" / "facebook",
    )
)
BASE_DIR = str(_MODEL_CACHE_ROOT / "dinov2-base")
SMALL_DIR = str(_MODEL_CACHE_ROOT / "dinov2-small")


def benchmark_torch(model_dir: str, tag: str, n_avg: int = 5) -> dict:
    """
    测 PyTorch CPU 推理延迟（batch=1 与 batch=2）。

    @author kk
    """
    proc = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval()
    dummy = Image.fromarray(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
    b1 = proc(images=dummy, return_tensors="pt")
    b2 = proc(images=[dummy, dummy], return_tensors="pt")

    def timeit(inputs, repeats):
        with torch.no_grad():
            for _ in range(2):
                model(**inputs)
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(**inputs)
        return (time.perf_counter() - t0) / repeats * 1000

    return {
        "tag": tag,
        "single_ms": round(timeit(b1, n_avg), 1),
        "double_ms": round(timeit(b2, n_avg), 1),
    }


def benchmark_onnx(onnx_path: str, tag: str, n_avg: int = 5) -> dict:
    """
    测 ONNX Runtime CPU 推理延迟（batch=1 与 batch=2）。

    @author kk
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 0  # 用满物理核
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    h, w = shape[2], shape[3]

    def timeit(batch: int, repeats: int) -> float:
        x = np.random.rand(batch, 3, h, w).astype(np.float32)
        for _ in range(2):
            sess.run(None, {name: x})
        t0 = time.perf_counter()
        for _ in range(repeats):
            sess.run(None, {name: x})
        return (time.perf_counter() - t0) / repeats * 1000

    return {
        "tag": tag,
        "single_ms": round(timeit(1, n_avg), 1),
        "double_ms": round(timeit(2, n_avg), 1),
    }


def export_onnx(model_dir: str, out_path: str, size: int = 224) -> None:
    """
    将 HF 模型导出为 ONNX（固定输入尺寸，仅 encoder 部分）。

    @author kk
    """
    model = AutoModel.from_pretrained(model_dir).eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}},
        opset_version=13, do_constant_folding=True,
    )


def main() -> None:
    """
    入口：依次跑 base/small 的 PyTorch 基准，ONNX 导出 + 动态 INT8 量化后重测。

    @author kk
    """
    import os

    results = []
    for name, d in [("base", BASE_DIR), ("small", SMALL_DIR)]:
        if not os.path.isdir(d):
            print(f"[跳过] {name} 权重不存在: {d}")
            continue
        results.append(benchmark_torch(d, f"PyTorch-FP32 {name}"))

        onnx_f32 = str(SCRIPT_DIR / f"{name}_fp32.onnx")
        onnx_i8 = str(SCRIPT_DIR / f"{name}_int8.onnx")
        if not os.path.exists(onnx_f32):
            print(f"[导出] {name} -> ONNX ...")
            export_onnx(d, onnx_f32)
        results.append(benchmark_onnx(onnx_f32, f"ONNX-FP32   {name}"))

        if not os.path.exists(onnx_i8):
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"[量化] {name} -> INT8 ...")
            # 仅量化 MatMul/Gemm（ViT 计算大头）；Conv 保持 FP32，规避 ConvInteger 无实现问题
            quantize_dynamic(
                onnx_f32, onnx_i8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
        results.append(benchmark_onnx(onnx_i8, f"ONNX-INT8   {name}"))

    print("\n===== 延迟基准（CPU，预热2次取5次均值）=====")
    print(f"{'变体':<22}{'单图 ms':>10}{'双图 ms':>10}")
    for r in results:
        print(f"{r['tag']:<22}{r['single_ms']:>10}{r['double_ms']:>10}")


if __name__ == "__main__":
    main()

"""
ONNX 工具 (onnx_tools.py)
==========================

- FusedTorch / OnnxFusedEngine : 融合引擎的 ONNX 版。
  ONNX 图里没有"按掩码取子集"的便宜写法，所以图里对整行所有像素计算，
  掩码和回写仍在 numpy 里做（与 OptimizedEngine 一致）。
- CnnWithScaler / build_cnn_module : 把 [StandardScaler -> 1D-CNN] 合成一个 torch 模块，
  导出 ONNX 时标准化也在图里，推理端不需要 sklearn。
- LabelOnlyProcessor : 只做背景掩码 + 分类的处理器，给 CNN 各版本用。
- quantize_linear_int8 : 对 MatMul/Gemm 做 int8 动态量化。
  只量化线性层，不量化 Conv：onnxruntime 的 CPU 后端对量化卷积的支持不完整。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from hsi.streaming.engine import BG_THRESHOLD, LineResult, _empty_result
from hsi.streaming.engine_opt import OptimizedEngine


class FusedTorch(nn.Module):
    """用 torch 重写融合引擎的数值部分（权重取自 OptimizedEngine），用于导出 ONNX。"""

    def __init__(self, eng: OptimizedEngine):
        super().__init__()
        for name in ("W", "bias", "loc", "prec"):
            self.register_buffer(name, torch.from_numpy(np.array(getattr(eng, name), np.float32)))

    def forward(self, x: torch.Tensor):
        z = x @ self.W + self.bias                    # (N, K+2)
        d = x - self.loc
        maha = ((d @ self.prec) * d).sum(dim=1)       # (N,) 马氏距离平方
        return z, maha


class CnnWithScaler(nn.Module):
    """输入 (N, B) 原始光谱 -> 标准化 -> 1D-CNN -> (N, 类别数) logits。"""

    def __init__(self, scaler, net: nn.Module):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(np.array(scaler.mean_, np.float32)))
        self.register_buffer("scale", torch.from_numpy(np.array(scaler.scale_, np.float32)))
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(((x - self.mean) / self.scale)[:, None, :])


def build_cnn_module(pipe):
    """pipe = make_baseline("cnn", "raw") 训练后的 Pipeline。返回 (module, classes)。"""
    scaler, clf = pipe[1], pipe[-1]
    return CnnWithScaler(scaler, clf.model_).eval(), clf.classes_


def export_onnx(module: nn.Module, example: torch.Tensor, path: Path,
                input_names: list[str], output_names: list[str]) -> Path:
    """导出 ONNX，第 0 维（像素数）设为动态，因为每行前景像素数不固定。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    torch.onnx.export(
        module, (example,), str(path),
        input_names=input_names, output_names=output_names,
        dynamic_axes={n: {0: "n"} for n in input_names + output_names},
        opset_version=17,
        dynamo=False,  # 用传统导出器：稳定，且不需要额外依赖 onnxscript
    )
    return path


def make_session(path: Path) -> ort.InferenceSession:
    """单线程、顺序执行的 CPU 会话：小矩阵上多线程只会增加抖动，也与 NumPy 基线公平。"""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def quantize_linear_int8(src: Path, dst: Path) -> Path:
    """对 MatMul / Gemm 做 int8 动态量化（权重离线量化，激活运行时量化）。"""
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    return dst


class OnnxFusedEngine:
    """融合引擎的 ONNX Runtime 版。图里算 (z, maha)，掩码和回写在 numpy 里。"""

    def __init__(self, path: Path, ref: OptimizedEngine):
        self.sess = make_session(path)
        self.k, self.classes, self.bg_threshold = ref.k, ref.classes, ref.bg_threshold

    def process_line(self, line: np.ndarray) -> LineResult:
        x = np.ascontiguousarray(line, dtype=np.float32)
        z, maha = self.sess.run(None, {"x": x})
        fg = z[:, -1] > self.bg_threshold
        out = _empty_result(x.shape[0])
        if fg.any():
            out.label[fg] = self.classes[np.argmax(z[fg][:, : self.k], axis=1)]
            out.moisture[fg] = z[fg][:, self.k]
            out.anomaly[fg] = maha[fg]
        return out


class LabelOnlyProcessor:
    """背景掩码 + 分类。logits_fn: (n, B) float32 -> (n, K) logits。"""

    def __init__(self, logits_fn, classes, bg_threshold: float = BG_THRESHOLD):
        self.logits_fn = logits_fn
        self.classes = np.asarray(classes, dtype=np.int8)
        self.bg_threshold = bg_threshold

    def process_line(self, line: np.ndarray) -> LineResult:
        x = np.ascontiguousarray(line, dtype=np.float32)
        fg = x.mean(axis=1) > self.bg_threshold
        out = _empty_result(x.shape[0])
        if fg.any():
            out.label[fg] = self.classes[np.argmax(self.logits_fn(x[fg]), axis=1)]
        return out


def torch_logits_fn(module: nn.Module):
    module.eval()

    def f(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return module(torch.from_numpy(x)).numpy()
    return f


def ort_logits_fn(sess: ort.InferenceSession):
    def f(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {"x": x})[0]
    return f

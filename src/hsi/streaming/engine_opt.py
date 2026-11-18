"""
优化版融合引擎 (engine_opt.py)
===============================

在 FusedEngine 基础上做三项优化，每项都可以单独在基准里验证收益：

1. 合并矩阵乘：分类得分 (K 列)、水分 (1 列)、行均值 (1 列，用于背景掩码)
   共用输入 x，拼成 (B, K+2) 的矩阵一次算完，少几次 numpy 调用。
2. dtype：float32。相机输出本来就是 float32，省掉每行的 float64 转换，内存也减半。
   代价是数值精度下降，马氏距离（精度矩阵条件数可能很大）尤其需要检查。
3. reuse_buffers：复用输出数组，省去每行 3 次分配。
   警告：开启后每次 process_line 返回的是同一批数组，调用方必须在下一次调用前用完
   或自己 copy，否则数据会被覆盖。

对外属性 W / bias / loc / prec / classes / k 也被 ONNX 导出复用。
"""

from __future__ import annotations

import numpy as np

from hsi.streaming.engine import BG_THRESHOLD, FusedEngine, LineResult, _empty_result


class OptimizedEngine:
    """接口与 FusedEngine 相同：process_line(line) -> LineResult。"""

    def __init__(self, base: FusedEngine, dtype=np.float32, reuse_buffers: bool = False):
        self.dtype = np.dtype(dtype)
        self.bg_threshold = base.bg_threshold if base.bg_threshold else BG_THRESHOLD
        self.reuse_buffers = reuse_buffers
        b, k = base.A_cls.shape
        self.k = k

        # 合并矩阵：前 k 列 = 分类得分，第 k 列 = 水分，最后一列 = 行均值（系数 1/B）
        w = np.zeros((b, k + 2))
        w[:, :k] = base.A_cls
        w[:, k] = base.A_reg
        w[:, k + 1] = 1.0 / b
        bias = np.zeros(k + 2)
        bias[:k] = base.b_cls
        bias[k] = base.b_reg           # 均值列的偏置为 0

        self.W = np.ascontiguousarray(w, dtype=self.dtype)
        self.bias = bias.astype(self.dtype)
        self.classes = base.classes
        self.loc = base.loc.astype(self.dtype)
        self.prec = np.ascontiguousarray(base.prec, dtype=self.dtype)
        self._buf: LineResult | None = None

    def _new_out(self, w: int) -> LineResult:
        if not self.reuse_buffers:
            return _empty_result(w)
        if self._buf is None or len(self._buf.label) != w:
            self._buf = _empty_result(w)
        else:  # 复用：重置为"全背景"状态
            self._buf.label.fill(0)
            self._buf.moisture.fill(np.nan)
            self._buf.anomaly.fill(np.nan)
        return self._buf

    def process_line(self, line: np.ndarray) -> LineResult:
        x = line if line.dtype == self.dtype else line.astype(self.dtype)
        z = x @ self.W + self.bias                 # (W, K+2)：对整行所有像素一次算完
        fg = z[:, -1] > self.bg_threshold          # 最后一列是行均值
        out = self._new_out(x.shape[0])
        if fg.any():
            zf = z[fg]
            out.label[fg] = self.classes[np.argmax(zf[:, : self.k], axis=1)]
            out.moisture[fg] = zf[:, self.k]
            d = x[fg] - self.loc
            out.anomaly[fg] = np.einsum("ij,ij->i", d @ self.prec, d)
        return out

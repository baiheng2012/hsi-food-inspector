"""
光谱预处理 (preprocessing.py)
==============================

提供三种常用的近红外光谱预处理，全部实现为 sklearn 兼容的 Transformer，
这样可以直接放进 sklearn.pipeline.Pipeline，在交叉验证中自动做到
"只用训练集 fit"，避免数据泄漏。

约定：输入 X 形状为 (N, B)，N 条光谱，每条 B 个波段；输出形状不变。

对外接口
--------
- snv(X)                     函数版 SNV
- msc_transform(X, ref)      函数版 MSC（需要给定参考光谱）
- SNV                        Transformer：无状态
- MSC                        Transformer：fit 时学习参考光谱（有状态，泄漏风险点）
- SavitzkyGolay              Transformer：平滑 / 求导
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-8  # 防止除以零


def _as_2d_float(X) -> np.ndarray:
    """转成二维 float32 数组；一维输入（单条光谱）会被当作 (1, B)。"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError(f"期望形状 (N, B)，实际得到 {X.shape}")
    return X


def snv(X) -> np.ndarray:
    """
    标准正态变量变换 (Standard Normal Variate)。

    对每条光谱：(x - 该光谱均值) / 该光谱标准差。
    - 逐行独立计算，不依赖其他样本，因此没有泄漏问题。
    - 对 x -> a*x + c（a>0）的变换不变，正好抵消乘性散射和加性偏移。
    """
    X = _as_2d_float(X)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + _EPS)


def msc_transform(X, reference: np.ndarray) -> np.ndarray:
    """
    多元散射校正 (Multiplicative Scatter Correction)，给定参考光谱。

    对每条光谱 x，最小二乘拟合 x ≈ a + b * reference，
    然后输出 (x - a) / b。下面用向量化公式一次算完所有行，避免 Python 循环：
        b = sum((x - mean(x)) * (ref - mean(ref))) / sum((ref - mean(ref))^2)
        a = mean(x) - b * mean(ref)
    """
    X = _as_2d_float(X)
    ref = np.asarray(reference, dtype=np.float32)
    ref_mean = ref.mean()
    ref_c = ref - ref_mean                           # 去中心化的参考光谱, (B,)
    x_mean = X.mean(axis=1, keepdims=True)           # (N, 1)
    b = ((X - x_mean) @ ref_c) / (ref_c @ ref_c + _EPS)   # (N,) 斜率
    b = np.where(np.abs(b) < _EPS, _EPS, b)          # 斜率接近 0 时避免除零
    a = x_mean[:, 0] - b * ref_mean                  # (N,) 截距
    return (X - a[:, None]) / b[:, None]


class SNV(BaseEstimator, TransformerMixin):
    """SNV 的 sklearn 封装。无状态：fit 什么也不学。"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return snv(X)


class MSC(BaseEstimator, TransformerMixin):
    """
    MSC 的 sklearn 封装。

    fit(X)      : 把训练集的平均光谱存为 reference_（这就是"状态"）
    transform(X): 用 reference_ 校正任意数据（训练集或测试集）

    放进 Pipeline 并做交叉验证时，sklearn 会保证每一折只用训练折 fit，
    测试折只调用 transform，从而不泄漏。
    """

    def fit(self, X, y=None):
        X = _as_2d_float(X)
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X):
        if not hasattr(self, "reference_"):
            raise RuntimeError("MSC 需要先 fit")
        return msc_transform(X, self.reference_)


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay 平滑 / 求导。无状态。

    参数
    ----
    window_length : 窗口宽度（波段数），必须为奇数，且大于 polyorder
    polyorder     : 窗口内拟合的多项式阶数
    deriv         : 0 = 只平滑；1 = 一阶导数（消除加性偏移）；2 = 二阶导数（再消除线性斜率）
    """

    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def _check(self, n_bands: int) -> None:
        if self.window_length % 2 == 0:
            raise ValueError("window_length 必须为奇数")
        if self.window_length <= self.polyorder:
            raise ValueError("window_length 必须大于 polyorder")
        if self.window_length > n_bands:
            raise ValueError("window_length 不能超过波段数")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_2d_float(X)
        self._check(X.shape[1])
        return savgol_filter(
            X, self.window_length, self.polyorder, deriv=self.deriv, axis=1
        ).astype(np.float32)

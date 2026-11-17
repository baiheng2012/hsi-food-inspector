"""
异常检测器 (anomaly.py)
========================

设定：只用"正常"光谱（新鲜/陈旧食品）训练，从未见过异物；测试时给每条光谱
打一个异常分数，分数越大越像异物。

三种检测器，统一接口（sklearn 风格）：
- fit(X_normal)        : 只用正常样本学习
- score_samples(X)     : 返回异常分数，越大越异常

注意：检测器本身不含预处理，请用 Pipeline 组合：
    make_detector("sam", prep="snv") = 预处理 -> 检测器
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline

from hsi.models.baselines import make_prep

DETECTOR_NAMES = ("sam", "mahalanobis", "iforest")
_EPS = 1e-8


class SAMDetector(BaseEstimator):
    """
    光谱角匹配 (Spectral Angle Mapper)。

    把光谱看成 B 维向量，计算它与"正常平均光谱"的夹角：
        angle = arccos( <x, ref> / (|x| |ref|) )
    夹角只取决于向量方向（光谱形状），不取决于长度，所以天然不受乘性散射影响。
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.reference_ = X.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        ref = self.reference_
        cos = (X @ ref) / (np.linalg.norm(X, axis=1) * np.linalg.norm(ref) + _EPS)
        return np.arccos(np.clip(cos, -1.0, 1.0))


class MahalanobisDetector(BaseEstimator):
    """
    马氏距离：考虑波段间相关性的"标准化距离"。
    光谱 100 个波段高度相关，直接估计协方差矩阵会病态，所以用 Ledoit-Wolf 收缩估计。
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.cov_ = LedoitWolf().fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X) -> np.ndarray:
        # sklearn 的 mahalanobis() 返回的是平方距离
        return self.cov_.mahalanobis(np.asarray(X, dtype=np.float64))


class IForestDetector(BaseEstimator):
    """Isolation Forest 封装：sklearn 原生分数越大越正常，这里取负号统一成"越大越异常"。"""

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        ).fit(X)
        self.n_features_in_ = X.shape[1]
        return self

    def score_samples(self, X) -> np.ndarray:
        return -self.model_.score_samples(np.asarray(X, dtype=np.float64))


def make_detector(name: str, prep: str = "raw"):
    """预处理 -> 检测器 的 Pipeline。注意 Pipeline 会把 score_samples 透传给最后一步。"""
    dets = {"sam": SAMDetector, "mahalanobis": MahalanobisDetector, "iforest": IForestDetector}
    if name not in dets:
        raise ValueError(f"未知检测器: {name}，可选 {DETECTOR_NAMES}")
    return make_pipeline(make_prep(prep), dets[name]())


def detection_rate_at_fpr(y_anomaly, scores, fpr: float = 0.01) -> float:
    """
    在"正常样本误报率 = fpr"时的异物检出率。
    阈值取正常样本分数的 (1 - fpr) 分位数，高于阈值判为异物。
    产线含义：好食品被误剔除的比例限制在 fpr 以内，能抓到多少异物。
    """
    y_anomaly = np.asarray(y_anomaly).astype(bool)
    scores = np.asarray(scores)
    thr = np.quantile(scores[~y_anomaly], 1.0 - fpr)
    return float((scores[y_anomaly] > thr).mean())

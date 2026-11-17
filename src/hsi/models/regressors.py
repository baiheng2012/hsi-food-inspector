"""
水分回归模型 (regressors.py)
=============================

- PLSReg            : 偏最小二乘回归（光谱定量分析的经典方法），sklearn 兼容
- make_regressor    : pls / ridge / gbm
- make_reg_baseline : 预处理 -> 标准化 -> 回归器
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsi.models.baselines import make_prep

REG_NAMES = ("pls", "ridge", "gbm")


class PLSReg(RegressorMixin, BaseEstimator):
    """PLS 回归封装：predict 返回一维数组（sklearn 原生返回 (N,1)，会让指标计算出错）。"""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.pls_ = PLSRegression(n_components=n_comp, scale=False).fit(X, np.asarray(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.pls_.predict(np.asarray(X, dtype=np.float64)).ravel()


def make_regressor(name: str):
    if name == "pls":
        return PLSReg(n_components=8)
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "gbm":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
    raise ValueError(f"未知回归器: {name}，可选 {REG_NAMES}")


def make_reg_baseline(model: str, prep: str):
    return make_pipeline(make_prep(prep), StandardScaler(), make_regressor(model))

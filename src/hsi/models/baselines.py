"""
分类基线模型 (baselines.py)
============================

提供：
- PLSDA            : 偏最小二乘判别分析（光谱学里的经典基线），sklearn 兼容
- make_prep(name)  : 按名字创建预处理方案
- make_classifier  : 按名字创建分类器（logreg / plsda / svm / rf / cnn）
- make_baseline    : 组合成 Pipeline = 预处理 -> 标准化 -> 分类器

统一成 sklearn Pipeline 的好处：交叉验证时每一折都只用训练折 fit，不泄漏；
之后做流式推理时，整条 Pipeline 可以整体保存和调用。
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hsi.models.cnn1d import TorchSpectralClassifier
from hsi.preprocessing import SNV, SavitzkyGolay

PREP_NAMES = ("raw", "snv", "sg1+snv", "sg2+snv")
MODEL_NAMES = ("logreg", "plsda", "svm", "rf", "cnn")


class PLSDA(ClassifierMixin, BaseEstimator):
    """
    PLS-DA：把类别做 one-hot 编码，用 PLS 回归去拟合，预测时取输出最大的那一类。

    为什么在光谱领域流行？光谱有上百个高度共线的波段，PLS 先把它们压缩成少数几个
    "潜变量"（latent variables）再回归，比直接做最小二乘稳定得多。

    参数
    ----
    n_components : 潜变量个数（相当于模型复杂度，太小欠拟合，太大过拟合）
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        # one-hot：(N, 类别数)，第 k 列表示"是不是第 k 类"
        Y = (y[:, None] == self.classes_[None, :]).astype(np.float64)
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        # scale=False：标准化已由 Pipeline 里的 StandardScaler 完成
        self.pls_ = PLSRegression(n_components=n_comp, scale=False).fit(X, Y)
        return self

    def predict(self, X):
        scores = self.pls_.predict(np.asarray(X, dtype=np.float64))  # (N, 类别数)
        return self.classes_[np.argmax(scores, axis=1)]


def make_prep(name: str):
    """
    预处理方案。
    raw     : 不处理
    snv     : 只做 SNV（抵消乘性散射 + 加性偏移，但抵消不了斜率漂移）
    sg1+snv : 一阶导（消除偏移）再 SNV（消除缩放）
    sg2+snv : 二阶导（消除偏移 + 线性斜率）再 SNV（消除缩放）
    """
    if name == "raw":
        return "passthrough"
    if name == "snv":
        return SNV()
    if name == "sg1+snv":
        return make_pipeline(SavitzkyGolay(11, 2, deriv=1), SNV())
    if name == "sg2+snv":
        return make_pipeline(SavitzkyGolay(11, 2, deriv=2), SNV())
    raise ValueError(f"未知预处理: {name}，可选 {PREP_NAMES}")


def make_classifier(name: str, epochs: int = 30):
    """按名字创建分类器。epochs 只对 cnn 有效。"""
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "plsda":
        return PLSDA(n_components=10)
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    if name == "cnn":
        return TorchSpectralClassifier(epochs=epochs, batch_size=128)
    raise ValueError(f"未知模型: {name}，可选 {MODEL_NAMES}")


def make_baseline(model: str, prep: str, epochs: int = 30):
    """完整流水线：预处理 -> 逐波段标准化 -> 分类器。"""
    return make_pipeline(make_prep(prep), StandardScaler(), make_classifier(model, epochs))

"""模型测试：守护 sklearn 兼容性、可复现性，以及"所有 模型×预处理 组合都能跑通"。"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from hsi.models.baselines import MODEL_NAMES, PLSDA, PREP_NAMES, make_baseline
from hsi.models.cnn1d import TorchSpectralClassifier
from hsi.preprocessing import SNV
from hsi.validation import object_level_accuracy


def _toy(n=120, b=40, seed=0):
    """三类正弦形状不同的光谱 + 小噪声，很容易分。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    x = np.linspace(0, 1, b)
    X = np.stack([np.sin((c + 1) * np.pi * x) for c in y]) + 0.05 * rng.normal(size=(n, b))
    return X.astype(np.float32), y


def test_plsda_learns_toy_problem():
    X, y = _toy()
    assert (PLSDA(5).fit(X, y).predict(X) == y).mean() > 0.9


def test_cnn_learns_toy_problem_and_proba_valid():
    X, y = _toy()
    clf = TorchSpectralClassifier(epochs=30, batch_size=32).fit(X, y)
    assert (clf.predict(X) == y).mean() > 0.85
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert clf.loss_history_[-1] < clf.loss_history_[0]  # 损失确实在下降


def test_cnn_reproducible_with_same_seed():
    X, y = _toy()
    a = TorchSpectralClassifier(epochs=3, batch_size=32, seed=1).fit(X, y).predict_proba(X)
    b = TorchSpectralClassifier(epochs=3, batch_size=32, seed=1).fit(X, y).predict_proba(X)
    assert np.allclose(a, b, atol=1e-4)


def test_cnn_is_clonable_and_works_in_pipeline():
    X, y = _toy()
    clone(TorchSpectralClassifier(epochs=2))  # sklearn 的 clone 要求 __init__ 参数规范
    pipe = make_pipeline(SNV(), TorchSpectralClassifier(epochs=2, batch_size=32)).fit(X, y)
    assert pipe.predict(X).shape == y.shape


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("prep", PREP_NAMES)
def test_every_model_prep_combination_runs(model, prep):
    # 防止"某个组合悄悄报错"（步骤 2 里 snv+sg1 的 bug 就是这类问题）
    X, y = _toy()
    pipe = make_baseline(model, prep, epochs=1).fit(X, y)
    assert pipe.predict(X).shape == y.shape


def test_object_level_accuracy_majority_vote():
    groups = np.array([1, 1, 1, 2, 2, 2])
    y_true = np.array([1, 1, 1, 2, 2, 2])
    y_pred = np.array([1, 1, 2, 2, 2, 1])  # 每个物体 3 个像素里错 1 个，投票后全对
    assert object_level_accuracy(groups, y_true, y_pred) == 1.0

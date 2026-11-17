"""验证工具的测试：重点钉住"分组划分无泄漏"这个契约。"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsi.data.simulator import SimConfig, generate_dataset
from hsi.preprocessing import SNV, SavitzkyGolay
from hsi.validation import build_pixel_table, evaluate, make_splitter


def _table():
    cfg = SimConfig(height=64, width=64, n_bands=40, n_items=6)
    return build_pixel_table(generate_dataset(6, cfg), per_item=30)


def test_pixel_table_caps_per_item_and_no_background():
    X, y, m, g = _table()
    assert X.shape[0] == y.shape[0] == m.shape[0] == g.shape[0]
    assert (y != 0).all()
    _, counts = np.unique(g, return_counts=True)
    assert counts.max() <= 30


def test_grouped_split_has_no_overlap():
    X, y, _, g = _table()
    for train, test in make_splitter("grouped", "classification").split(X, y, g):
        assert set(g[train]).isdisjoint(g[test])


def test_random_split_does_leak():
    # 反面教材：随机划分确实会让同一物体同时出现在训练集和测试集
    X, y, _, g = _table()
#    train, test = next(make_splitter("random", "classification").split(X, y, g))
    train, test = next(make_splitter("random", "classification").split(X, y))
    assert len(set(g[train]) & set(g[test])) > 0


def test_unknown_scheme_raises():
    import pytest
    with pytest.raises(ValueError):
        make_splitter("bogus", "classification")

def test_nested_pipeline_with_stateless_steps_scores_finite():
    # 守护：嵌套 Pipeline + 无状态变换器不能悄悄产生 NaN
    X, y, _, g = _table()
    inner = make_pipeline(SNV(), SavitzkyGolay(11, 2, deriv=1))
    pipe = make_pipeline(inner, StandardScaler(), LogisticRegression(max_iter=500))
    scores = evaluate(pipe, X, y, g, "grouped", "classification", "f1_macro", n_jobs=1)
    assert np.isfinite(scores).all()
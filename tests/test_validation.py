"""验证工具的测试：重点钉住"分组划分无泄漏"这个契约。"""

import numpy as np

from hsi.data.simulator import SimConfig, generate_dataset
from hsi.validation import build_pixel_table, make_splitter


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
    train, test = next(make_splitter("random", "classification").split(X, y, g))
    assert len(set(g[train]) & set(g[test])) > 0


def test_unknown_scheme_raises():
    import pytest
    with pytest.raises(ValueError):
        make_splitter("bogus", "classification")

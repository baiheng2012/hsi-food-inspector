"""
模拟器的单元测试。
运行：pytest -v
测试的意义：保证"可复现"、"形状正确"、"标签合法"，后续所有模块都依赖这些性质。
"""

import numpy as np

from hsi.data.simulator import (
    BACKGROUND,
    FOREIGN,
    HARD_CONFIG,
    SUBTLE_CONFIG,
    SimConfig,
    base_spectrum,
    generate_dataset,
    generate_scene,
    make_wavelengths,
)


def test_shapes_and_dtypes():
    cfg = SimConfig(height=64, width=48, n_bands=50)
    s = generate_scene(cfg, 0)
    assert s.cube.shape == (64, 48, 50)
    assert s.cube.dtype == np.float32
    assert s.label_map.shape == (64, 48)
    assert s.moisture_map.shape == (64, 48)
    assert s.item_map.shape == (64, 48)
    assert s.wavelengths.shape == (50,)


def test_reproducible():
    cfg = SimConfig()
    a = generate_scene(cfg, 3)
    b = generate_scene(cfg, 3)
    assert np.array_equal(a.cube, b.cube)  # 相同种子 -> 完全一致


def test_different_scenes_differ():
    cfg = SimConfig()
    a = generate_scene(cfg, 0)
    b = generate_scene(cfg, 1)
    assert not np.array_equal(a.cube, b.cube)


def test_labels_valid_and_background_exists():
    s = generate_scene(SimConfig(), 0)
    assert set(np.unique(s.label_map)).issubset({0, 1, 2, 3})
    assert (s.label_map == BACKGROUND).any()


def test_moisture_only_on_food():
    s = generate_scene(SimConfig(n_items=12), 0)
    # 只有新鲜/陈旧食品有水分值；背景和异物为 NaN
    has_moisture = ~np.isnan(s.moisture_map)
    labels_with_m = np.unique(s.label_map[has_moisture])
    assert set(labels_with_m).issubset({1, 2})
    assert np.isnan(s.moisture_map[s.label_map == FOREIGN]).all()


def test_item_ids_unique_across_scenes():
    ds = generate_dataset(3, SimConfig())
    ids = [set(np.unique(s.item_map)) - {0} for s in ds]
    assert ids[0].isdisjoint(ids[1]) and ids[1].isdisjoint(ids[2])


def test_flatten_foreground():
    s = generate_scene(SimConfig(), 0)
    x, y, m, g = s.flatten(foreground_only=True)
    assert x.shape[0] == y.shape[0] == m.shape[0] == g.shape[0]
    assert (y != BACKGROUND).all()


def test_water_absorption_physics():
    # 水分越高，1450nm 附近的反射率应越低（吸收越深）
    wl = make_wavelengths(SimConfig())
    idx = int(np.argmin(np.abs(wl - 1450)))
    low = base_spectrum(wl, 1, 0.3)[idx]
    high = base_spectrum(wl, 1, 0.9)[idx]
    assert high < low

def test_hard_mode_changes_spectra_but_not_geometry():
    # 困难模式只改光谱，不改物体的位置/类别（随机数抽取顺序没变）
    easy = generate_scene(SimConfig(), 0)
    hard = generate_scene(HARD_CONFIG, 0)
    assert np.array_equal(easy.label_map, hard.label_map)
    assert np.array_equal(easy.item_map, hard.item_map)
    assert not np.array_equal(easy.cube, hard.cube)

def test_subtle_foreign_is_closer_to_food():
    # 难检异物的平均光谱与新鲜食品的距离，应小于普通异物
    def foreign_to_fresh_dist(cfg):
        s = generate_scene(cfg, 0)
        fg = s.cube[s.label_map == FOREIGN].mean(axis=0)
        fr = s.cube[s.label_map == 1].mean(axis=0)
        return np.linalg.norm(fg - fr)

    assert foreign_to_fresh_dist(SUBTLE_CONFIG) < foreign_to_fresh_dist(HARD_CONFIG)

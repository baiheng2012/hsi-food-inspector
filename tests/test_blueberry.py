"""蓝莓加载器的纯函数测试（不需要真实数据）。"""

import numpy as np

from hsi.data.blueberry import bin_bands, otsu, segment_berries


def test_bin_bands_shape_and_wavelengths():
    wl = np.linspace(400, 1000, 301).astype(np.float32)
    X = np.random.default_rng(0).random((5, 301)).astype(np.float32)
    xb, wlb = bin_bands(X, wl, 450, 950, 4)
    assert xb.shape[0] == 5 and xb.shape[1] == len(wlb)
    assert wlb.min() >= 450 and wlb.max() <= 950


def test_bin_bands_averages_neighbours():
    wl = np.arange(8, dtype=np.float32) + 500
    X = np.arange(8, dtype=np.float32)[None, :]
    xb, _ = bin_bands(X, wl, 500, 510, 4)
    assert np.allclose(xb, [[1.5, 5.5]])

"""
def test_otsu_separates_two_modes():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.1, 0.01, 1000), rng.normal(0.5, 0.02, 1000)]
    assert 0.2 < otsu(x) < 0.4
"""
def test_otsu_separates_two_modes():
    rng = np.random.default_rng(0)
    low = rng.normal(0.1, 0.01, 1000)
    high = rng.normal(0.5, 0.02, 1000)
    t = otsu(np.r_[low, high])
    # 任何落在两个峰之间空隙里的阈值都是有效的大津阈值，这里只要求它能完美分开两类
    assert (low <= t).all() and (high > t).all()

def test_segment_finds_disks_and_drops_edge_objects():
    yy, xx = np.mgrid[0:120, 0:120]
    img = np.full((120, 120), 0.08)
    for cy, cx in ((30, 30), (30, 80), (80, 30), (80, 80)):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 100] = 0.5
    img[0:8, 100:120] = 0.5                       # 触边的碎片，应被剔除
    lab, _ = segment_berries(img, np.ones_like(img, dtype=bool))
    assert lab.max() == 4

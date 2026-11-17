"""异常检测测试：守护"形状不变性""只用正常样本训练""指标计算正确"。"""

import numpy as np
import pytest

from hsi.models.anomaly import (
    DETECTOR_NAMES,
    SAMDetector,
    detection_rate_at_fpr,
    make_detector,
)


def _data(n=200, b=30, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, b)
    normal = np.sin(2 * np.pi * x) + 2 + 0.05 * rng.normal(size=(n, b))
    odd = np.cos(3 * np.pi * x) + 2 + 0.05 * rng.normal(size=(50, b))
    return normal.astype(np.float32), odd.astype(np.float32)


def test_sam_invariant_to_scale():
    # SAM 的核心承诺：整体乘以常数，分数不变
    normal, _ = _data()
    det = SAMDetector().fit(normal)
    assert np.allclose(det.score_samples(normal), det.score_samples(3.7 * normal), atol=1e-6)


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_detectors_rank_anomalies_higher(name):
    normal, odd = _data()
    det = make_detector(name).fit(normal)
    assert det.score_samples(odd).mean() > det.score_samples(normal).mean()


def test_detection_rate_perfect_and_chance():
    y = np.array([0] * 100 + [1] * 20)
    assert detection_rate_at_fpr(y, np.r_[np.zeros(100), np.ones(20)], 0.01) == 1.0
    rng = np.random.default_rng(0)
    rate = detection_rate_at_fpr(y, rng.random(120), 0.1)
    assert rate < 0.5  # 随机分数时检出率应接近 fpr，远低于 0.5


def test_unknown_detector_raises():
    with pytest.raises(ValueError):
        make_detector("bogus")

"""
流式推理测试4。
模型在小数据上训练一次（module 级 fixture），所有测试共用。
"""

import dataclasses

import numpy as np
import pytest

from hsi.data.simulator import HARD_CONFIG, generate_scene
from hsi.streaming.engine import FusedEngine, SklearnProcessor, fit_models
from hsi.streaming.linescan import iter_lines

PARAMS = {
    "data": {"config": "hard", "n_scenes": 6, "per_item": 30, "seed": 0},
    "models": {
        "classifier": {"name": "plsda", "prep": "raw"},
        "regressor": {"name": "ridge", "prep": "raw"},
        "detector": {"name": "mahalanobis", "prep": "raw"},
    },
}


@pytest.fixture(scope="module")
def models():
    return fit_models(PARAMS)


@pytest.fixture(scope="module")
def scene():
    return generate_scene(dataclasses.replace(HARD_CONFIG, seed=500), 0)


def test_iter_lines_covers_cube(scene):
    lines = list(iter_lines(scene.cube))
    assert len(lines) == scene.cube.shape[0]
    assert lines[0].shape == scene.cube.shape[1:]
    assert np.array_equal(np.stack(lines), scene.cube)


def test_iter_lines_rejects_bad_shape():
    with pytest.raises(ValueError):
        next(iter_lines(np.zeros((4, 5))))


def test_fused_matches_sklearn(models, scene):
    fused, sk = FusedEngine.from_sklearn(models), SklearnProcessor(models)
    for ln in iter_lines(scene.cube):
        a, b = fused.process_line(ln), sk.process_line(ln)
        assert np.array_equal(a.label, b.label)
        assert np.allclose(a.moisture, b.moisture, atol=1e-6, equal_nan=True)
        assert np.allclose(a.anomaly, b.anomaly, rtol=1e-6, equal_nan=True)


def test_background_line_is_skipped(models):
    fused = FusedEngine.from_sklearn(models)
    line = np.full((16, 100), 0.05, dtype=np.float32)   # 全是暗背景
    r = fused.process_line(line)
    assert (r.label == 0).all() and np.isnan(r.moisture).all() and np.isnan(r.anomaly).all()


def test_foreground_pixels_get_labels(models, scene):
    # 场景里有食品，融合引擎应当把大部分前景像素判为非背景
    fused = FusedEngine.from_sklearn(models)
    hit = tot = 0
    for i, ln in enumerate(iter_lines(scene.cube)):
        truth_fg = scene.label_map[i] != 0
        hit += int((fused.process_line(ln).label[truth_fg] != 0).sum())
        tot += int(truth_fg.sum())
    assert hit / tot > 0.98


def test_unsupported_prep_raises():
    params = {**PARAMS, "models": {**PARAMS["models"],
                                   "regressor": {"name": "ridge", "prep": "snv"}}}
    with pytest.raises(NotImplementedError):
        FusedEngine.from_sklearn(fit_models(params))

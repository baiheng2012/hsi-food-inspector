"""
优化与 ONNX 测试：守护"每一种优化都不改变（或只在可接受范围内改变）结果"。
"""

import dataclasses

import numpy as np
import pytest
import torch
from sklearn.pipeline import Pipeline  # noqa: F401  (保持与其他测试一致的依赖显式)

from hsi.data.simulator import HARD_CONFIG, generate_scene
from hsi.models.baselines import make_baseline
from hsi.streaming.engine import FusedEngine, fit_models
from hsi.streaming.engine_opt import OptimizedEngine
from hsi.streaming.linescan import iter_lines
from hsi.streaming.onnx_tools import (
    FusedTorch,
    OnnxFusedEngine,
    build_cnn_module,
    export_onnx,
    make_session,
)

PARAMS = {
    "data": {"config": "hard", "n_scenes": 6, "per_item": 30, "seed": 0},
    "models": {
        "classifier": {"name": "plsda", "prep": "raw"},
        "regressor": {"name": "ridge", "prep": "raw"},
        "detector": {"name": "mahalanobis", "prep": "raw"},
    },
}


@pytest.fixture(scope="module")
def base():
    return FusedEngine.from_sklearn(fit_models(PARAMS))


@pytest.fixture(scope="module")
def scene():
    return generate_scene(dataclasses.replace(HARD_CONFIG, seed=500), 0)


def test_opt_f64_matches_fused(base, scene):
    opt = OptimizedEngine(base, np.float64)
    for ln in iter_lines(scene.cube):
        a, b = base.process_line(ln), opt.process_line(ln)
        assert np.array_equal(a.label, b.label)
        assert np.allclose(a.moisture, b.moisture, atol=1e-9, equal_nan=True)
        assert np.allclose(a.anomaly, b.anomaly, rtol=1e-9, equal_nan=True)


def test_opt_f32_close_to_fused(base, scene):
    # float32 允许微小差异。若此测试失败，说明 float32 对本模型不安全，这本身是要写进报告的发现
    opt = OptimizedEngine(base, np.float32)
    same = total = 0
    for ln in iter_lines(scene.cube):
        a, b = base.process_line(ln), opt.process_line(ln)
        same += int((a.label == b.label).sum())
        total += len(a.label)
        both = np.isfinite(a.moisture) & np.isfinite(b.moisture)
        assert np.allclose(a.moisture[both], b.moisture[both], atol=1e-2)
    assert same / total > 0.999


def test_reuse_buffers_same_values_but_shared_arrays(base, scene):
    plain = OptimizedEngine(base, np.float32)
    reuse = OptimizedEngine(base, np.float32, reuse_buffers=True)
    lines = list(iter_lines(scene.cube))
    r1 = reuse.process_line(lines[0])
    assert np.array_equal(r1.label, plain.process_line(lines[0]).label)
    r2 = reuse.process_line(lines[1])
    assert r1.label is r2.label  # 确认缓冲确实被复用（这也是它的风险所在）


def test_fused_torch_matches_numpy(base, scene):
    opt = OptimizedEngine(base, np.float32)
    x = scene.cube[10].astype(np.float32)
    z, m = FusedTorch(opt)(torch.from_numpy(x))
    assert np.allclose(z.numpy(), x @ opt.W + opt.bias, atol=1e-4)
    d = x - opt.loc
    assert np.allclose(m.numpy(), np.einsum("ij,ij->i", d @ opt.prec, d), rtol=1e-3)


def test_onnx_fused_matches_opt(base, scene, tmp_path):
    opt = OptimizedEngine(base, np.float32)
    path = export_onnx(FusedTorch(opt), torch.zeros(4, opt.W.shape[0]),
                       tmp_path / "f.onnx", ["x"], ["z", "maha"])
    onnx_eng = OnnxFusedEngine(path, opt)
    for ln in iter_lines(scene.cube):
        a, b = opt.process_line(ln), onnx_eng.process_line(ln)
        assert np.array_equal(a.label, b.label)
        assert np.allclose(a.moisture, b.moisture, atol=1e-4, equal_nan=True)


def test_cnn_onnx_logits_match_torch(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40)).astype(np.float32)
    y = rng.integers(1, 4, 60)
    pipe = make_baseline("cnn", "raw", epochs=1).fit(X, y)
    module, _ = build_cnn_module(pipe)
    path = export_onnx(module, torch.zeros(4, 40), tmp_path / "c.onnx", ["x"], ["logits"])
    got = make_session(path).run(None, {"x": X[:7]})[0]
    with torch.no_grad():
        want = module(torch.from_numpy(X[:7])).numpy()
    assert np.allclose(got, want, atol=1e-4)

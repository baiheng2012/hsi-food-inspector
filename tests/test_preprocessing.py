"""预处理的测试：守护数学性质（不变性、状态、参数检查）。"""

import numpy as np
import pytest

from hsi.preprocessing import MSC, SNV, SavitzkyGolay, snv


def _spectra(n=50, b=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, b)
    ref = 0.5 + 0.2 * np.sin(2 * np.pi * x)
    return ref, ref + 0.02 * rng.normal(size=(n, b))


def test_snv_zero_mean_unit_std():
    _, X = _spectra()
    out = snv(X)
    assert np.allclose(out.mean(axis=1), 0, atol=1e-4)
    assert np.allclose(out.std(axis=1), 1, atol=1e-3)


def test_snv_invariant_to_scale_and_offset():
    # SNV 的核心承诺：乘性散射 + 加性偏移不影响输出
    _, X = _spectra()
    assert np.allclose(snv(X), snv(1.7 * X + 0.3), atol=1e-3)


def test_msc_removes_scatter():
    # 构造 x_i = a_i + b_i * ref，MSC 后所有光谱应几乎相同
    ref, _ = _spectra()
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.05, size=(40, 1))
    b = rng.uniform(0.7, 1.3, size=(40, 1))
    X = (a + b * ref).astype(np.float32)
    out = MSC().fit_transform(X)
    assert out.std(axis=0).max() < 1e-3


def test_msc_reference_learned_from_fit_only():
    # 泄漏防线：reference_ 只取决于 fit 时给的数据
    _, A = _spectra(seed=0)
    _, B = _spectra(seed=5)
    m = MSC().fit(A)
    m.transform(B)  # 对其他数据 transform 不应改变状态
    assert np.allclose(m.reference_, A.astype(np.float32).mean(axis=0))


def test_msc_requires_fit():
    with pytest.raises(RuntimeError):
        MSC().transform(np.ones((2, 10)))


def test_savgol_deriv1_removes_offset():
    _, X = _spectra()
    sg = SavitzkyGolay(11, 2, deriv=1)
    assert np.allclose(sg.transform(X), sg.transform(X + 0.4), atol=1e-4)


def test_savgol_bad_params_raise():
    X = np.ones((3, 30))
    with pytest.raises(ValueError):
        SavitzkyGolay(window_length=10).transform(X)  # 偶数
    with pytest.raises(ValueError):
        SavitzkyGolay(window_length=3, polyorder=3).transform(X)  # 窗口 <= 阶数


def test_transformers_keep_shape():
    _, X = _spectra()
    for t in (SNV(), MSC(), SavitzkyGolay()):
        assert t.fit_transform(X).shape == X.shape

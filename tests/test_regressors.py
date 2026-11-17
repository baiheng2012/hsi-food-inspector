import numpy as np
import pytest

from hsi.models.regressors import REG_NAMES, make_reg_baseline


def _toy(n=150, b=30, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.3, 0.9, n)
    x = np.linspace(0, 1, b)
    X = 0.6 - y[:, None] * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2) + 0.01 * rng.normal(size=(n, b))
    return X.astype(np.float32), y


@pytest.mark.parametrize("model", REG_NAMES)
@pytest.mark.parametrize("prep", ["raw", "snv", "sg1+snv"])
def test_every_regressor_runs_and_predict_is_1d(model, prep):
    X, y = _toy()
    pred = make_reg_baseline(model, prep).fit(X, y).predict(X)
    assert pred.shape == y.shape  # 必须是一维，防止 PLS 的 (N,1) 陷阱


def test_pls_fits_toy_problem():
    X, y = _toy()
    pred = make_reg_baseline("pls", "raw").fit(X, y).predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.95

from hsi.regression_check import compare_to_baseline

BASE = {"cls_f1_macro": 0.90, "reg_rmse_fresh": 0.030}


def test_identical_metrics_pass():
    assert compare_to_baseline(dict(BASE), BASE) == []


def test_higher_is_better_metric_drop_fails():
    cur = {"cls_f1_macro": 0.80, "reg_rmse_fresh": 0.030}
    fails = compare_to_baseline(cur, BASE)
    assert len(fails) == 1 and "cls_f1_macro" in fails[0]


def test_lower_is_better_metric_rise_fails():
    # RMSE 变大是退化，方向和 F1 相反
    cur = {"cls_f1_macro": 0.90, "reg_rmse_fresh": 0.060}
    fails = compare_to_baseline(cur, BASE)
    assert len(fails) == 1 and "reg_rmse_fresh" in fails[0]


def test_small_fluctuation_within_tolerance_passes():
    cur = {"cls_f1_macro": 0.89, "reg_rmse_fresh": 0.0305}
    assert compare_to_baseline(cur, BASE) == []


def test_missing_metric_fails():
    assert compare_to_baseline({"cls_f1_macro": 0.9}, BASE) != []

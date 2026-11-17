"""
统一评估入口 (evaluation.py)
=============================

把"生成数据 -> 三个任务的分组交叉验证 -> 汇总指标"封装成函数，供三处复用：
- scripts/prepare_data.py : 调用 build_data 生成像素表
- scripts/train.py        : 调用 evaluate_all 得到指标并记录
- tests/test_model_regression.py : 调用 run_evaluation，与基线比较

输出的指标是扁平字典 {指标名: 浮点数}，便于写 JSON、记录到 MLflow、比较基线。
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score

from hsi.data.simulator import (
    AGED,
    FOREIGN,
    FRESH,
    HARD_CONFIG,
    SUBTLE_CONFIG,
    generate_dataset,
)
from hsi.models.anomaly import detection_rate_at_fpr, make_detector
from hsi.models.baselines import make_baseline
from hsi.models.regressors import make_reg_baseline
from hsi.validation import (
    build_pixel_table,
    object_level_accuracy,
    oof_anomaly_scores,
    oof_predict,
    oof_regress,
)

SIM_CONFIGS = {"hard": HARD_CONFIG, "subtle": SUBTLE_CONFIG}


def build_data(data_params: dict):
    """
    按 params["data"] 生成像素表。
    返回 X (N,B), y (N,) 类别, moisture (N,) 水分, groups (N,) 物体 ID
    """
    base = SIM_CONFIGS[data_params["config"]]
    cfg = dataclasses.replace(base, seed=data_params["seed"])  # SimConfig 是 frozen，用 replace 改种子
    scenes = generate_dataset(data_params["n_scenes"], cfg)
    return build_pixel_table(scenes, per_item=data_params["per_item"], seed=data_params["seed"])


def evaluate_all(X, y, moisture, groups, params: dict) -> dict:
    """
    在给定像素表上评估三个任务，全部使用按物体分组的交叉验证。

    返回的指标：
    cls_f1_macro            分类：像素级 macro-F1（折间均值）
    cls_object_acc          分类：物体级准确率（像素投票）
    reg_rmse_fresh          回归：新鲜食品类内 RMSE（越小越好）
    reg_r2_fresh            回归：新鲜食品类内 R²
    anom_auc                异物检测：ROC-AUC
    anom_detect_at_1pct_fpr 异物检测：1% 误报率下的检出率
    """
    cv = params["cv"]
    mdl = params["models"]
    n_splits, seed = cv["n_splits"], cv["seed"]
    out: dict[str, float] = {}

    # ---- 1) 分类 ----
    clf = make_baseline(mdl["classifier"]["name"], mdl["classifier"]["prep"])
    pred, splits = oof_predict(clf, X, y, groups, n_splits=n_splits, seed=seed)
    folds = [f1_score(y[te], pred[te], average="macro") for _, te in splits]
    out["cls_f1_macro"] = float(np.mean(folds))
    out["cls_object_acc"] = object_level_accuracy(groups, y, pred)

    # ---- 2) 水分回归：只用食品像素，并只在"新鲜"类内报告（真正的水分预测能力）----
    food = np.isin(y, (FRESH, AGED))
    reg = make_reg_baseline(mdl["regressor"]["name"], mdl["regressor"]["prep"])
    rpred, _ = oof_regress(reg, X[food], moisture[food], groups[food],
                           n_splits=n_splits, seed=seed)
    fresh = y[food] == FRESH
    m_f, p_f = moisture[food][fresh], rpred[fresh]
    out["reg_rmse_fresh"] = float(np.sqrt(mean_squared_error(m_f, p_f)))
    out["reg_r2_fresh"] = float(r2_score(m_f, p_f))

    # ---- 3) 异物检测：只用正常样本训练 ----
    det = make_detector(mdl["detector"]["name"], mdl["detector"]["prep"])
    scores = oof_anomaly_scores(det, X, y, groups, n_splits=n_splits, seed=seed)
    is_foreign = y == FOREIGN
    out["anom_auc"] = float(roc_auc_score(is_foreign, scores))
    out["anom_detect_at_1pct_fpr"] = detection_rate_at_fpr(is_foreign, scores, 0.01)
    return out


def run_evaluation(params: dict) -> dict:
    """从头跑一遍：生成数据 + 评估。回归测试用它。"""
    X, y, m, g = build_data(params["data"])
    return evaluate_all(X, y, m, g, params)

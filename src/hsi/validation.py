"""
验证工具 (validation.py)
=========================

目标：让"评估方式"成为一等公民，并且能直接量化数据泄漏的影响。

两种划分方案
-----------
- "random"  : 像素级随机划分。同一物体的像素会同时落在训练集和测试集（有泄漏）。
- "grouped" : 按物体 ID 分组划分。同一物体的所有像素只在训练集或只在测试集（正确做法）。

对外接口
--------
- build_pixel_table(scenes, per_item, seed) : 多个场景 -> 像素表，并对每个物体限量采样
- make_splitter(scheme, task, ...)           : 创建 sklearn 划分器
- assert_no_group_overlap(splits, groups)    : 断言训练/测试没有共同物体
- evaluate(estimator, X, y, groups, ...)     : 交叉验证并返回每折得分
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from hsi.data.simulator import Scene


def build_pixel_table(scenes: list[Scene], per_item: int = 100, seed: int = 0):
    """
    把多个场景合并成像素表，只保留前景（食品 + 异物）。

    为什么要对每个物体限量采样？
    1) 一个物体有上千个像素，高度相关，全用既慢又没必要；
    2) 避免大物体在指标里占过大权重。

    返回
    ----
    X (N,B), y (N,) 类别, moisture (N,) 水分(异物为 NaN), groups (N,) 物体 ID
    """
    rng = np.random.default_rng(seed)
    xs, ys, ms, gs = [], [], [], []
    for scene in scenes:
        x, y, m, g = scene.flatten(foreground_only=True)
        for gid in np.unique(g):
            idx = np.flatnonzero(g == gid)
            if len(idx) > per_item:
                idx = rng.choice(idx, size=per_item, replace=False)
            xs.append(x[idx]); ys.append(y[idx]); ms.append(m[idx]); gs.append(g[idx])
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ms), np.concatenate(gs))


def make_splitter(scheme: str, task: str, n_splits: int = 5, seed: int = 0):
    """
    创建划分器。
    scheme: "random" | "grouped"
    task  : "classification"（需要按类别分层） | "regression"
    """
    if scheme == "random":
        if task == "classification":
            return StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        return KFold(n_splits, shuffle=True, random_state=seed)
    if scheme == "grouped":
        if task == "classification":
            return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        return GroupKFold(n_splits)  # 回归：按组划分（确定性）
    raise ValueError(f"未知 scheme: {scheme}")


def assert_no_group_overlap(splits, groups: np.ndarray) -> None:
    """断言每一折的训练集与测试集没有共同的物体 ID。出现重叠就是泄漏。"""
    for train_idx, test_idx in splits:
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        assert not overlap, f"发现数据泄漏：{len(overlap)} 个物体同时在训练集和测试集"


def evaluate(estimator, X, y, groups, scheme: str, task: str, scoring: str,
             n_splits: int = 5, seed: int = 0, n_jobs: int = -1) -> np.ndarray:
    """
    对 estimator（通常是 Pipeline）做交叉验证，返回每一折的得分数组。
    scheme="grouped" 时会自动断言没有泄漏。
    error_score="raise"：任何一折出错都直接抛异常，杜绝"静默变成 NaN"。
    """
    splitter = make_splitter(scheme, task, n_splits, seed)
    # random 方案的划分器不使用 groups，传了会产生警告，所以只在 grouped 时传
    split_groups = groups if scheme == "grouped" else None
    splits = list(splitter.split(X, y, split_groups))
    if scheme == "grouped":
        assert_no_group_overlap(splits, groups)
    return cross_val_score(estimator, X, y, cv=splits, scoring=scoring,
                           n_jobs=n_jobs, error_score="raise")

def oof_predict(estimator, X, y, groups, scheme: str = "grouped",
                task: str = "classification", n_splits: int = 5,
                seed: int = 0, n_jobs: int = 1):
    """
    交叉验证并返回"折外预测"(out-of-fold)：每个样本的预测值都来自
    "没见过该样本（该物体）"的模型。

    返回 (pred, splits)：
    pred   : (N,) 每个像素的预测类别
    splits : 划分列表，可用来计算每一折的得分
    """
    splitter = make_splitter(scheme, task, n_splits, seed)
    split_groups = groups if scheme == "grouped" else None
    splits = list(splitter.split(X, y, split_groups))
    if scheme == "grouped":
        assert_no_group_overlap(splits, groups)
    pred = cross_val_predict(estimator, X, y, cv=splits, n_jobs=n_jobs)
    return pred, splits


def object_level_accuracy(groups, y_true, y_pred) -> float:
    """
    物体级准确率：同一物体的所有像素预测"投票"，多数票作为该物体的预测，
    再与物体真实类别比较。产线上最终是对物体做剔除/放行决策，这个指标更贴近业务。
    """
    df = pd.DataFrame({"g": groups, "t": y_true, "p": y_pred})
    voted = df.groupby("g")["p"].agg(lambda s: s.mode().iloc[0])
    truth = df.groupby("g")["t"].first()  # 同一物体的真实类别是常数
    return float((voted == truth).mean())
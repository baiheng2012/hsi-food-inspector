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
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
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
    """
    splitter = make_splitter(scheme, task, n_splits, seed)
    splits = list(splitter.split(X, y, groups))  # 先生成所有划分，便于检查
    if scheme == "grouped":
        assert_no_group_overlap(splits, groups)
    return cross_val_score(estimator, X, y, cv=splits, scoring=scoring, n_jobs=n_jobs)

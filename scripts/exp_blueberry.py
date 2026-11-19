"""
真实数据实验：只用好果训练，检测缺陷果（果级评估）。

1) 托盘阴性对照：托盘本应与好坏无关。若能从托盘像素区分"这是 Good 还是 Bad 文件的拍摄"，
   说明两次拍摄存在批次效应，模型可能学到拍摄差异而不是缺陷。
2) 检测实验：检测器 x 预处理 x 是否做托盘归一化。
   每折：只用训练折里的好果像素 fit；测试折的所有果打分；
   果分数 = 该果像素异常分数的第 90 百分位。
   指标：果级 AUC(+bootstrap 95% 区间)、5% 误报率下的检出率（只有 42 颗好果，这个点很粗）。

运行：uv run python scripts/exp_blueberry.py
输出：docs/results/blueberry_detection.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsi.models.anomaly import DETECTOR_NAMES, make_detector

PREPS = ("raw", "snv", "sg1+snv")
FPR = 0.05


def tray_control(d: dict) -> float:
    """托盘上预测"来自哪个文件"：上半幅训练、下半幅测试（避免空间相邻像素泄漏）。AUC≈0.5 为无批次效应。"""
    X, s, row = d["X_tray"], d["tray_source"], d["tray_row"]
    cut = np.median(row)
    tr, te = row <= cut, row > cut
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X[tr], s[tr])
    return float(roc_auc_score(s[te], clf.predict_proba(X[te])[:, 1]))


def berry_scores(X, y, groups, detector, seed=0, n_splits=5, q=90):
    """果级折外异常分数。返回 (果ID, 果标签, 果分数)。"""
    ids = np.unique(groups)
    lab = np.array([y[groups == i][0] for i in ids])
    scores = np.zeros(len(ids))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(ids, lab):
        train_px = np.isin(groups, ids[tr]) & (y == 0)       # 只用训练折的好果
        det = clone(detector).fit(X[train_px])
        for j in te:
            scores[j] = np.percentile(det.score_samples(X[groups == ids[j]]), q)
    return ids, lab, scores


def tpr_at_fpr(lab, scores, fpr=FPR) -> float:
    thr = np.quantile(scores[lab == 0], 1 - fpr)
    return float((scores[lab == 1] > thr).mean())


def bootstrap_auc(lab, scores, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
    out = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        out.append(roc_auc_score(lab[idx], scores[idx]))
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    d = dict(np.load("data/blueberry/processed/blueberry.npz"))
    X, y, g = d["X"], d["y"], d["groups"]
    print(f"像素 {len(X)}，波段 {X.shape[1]}，果 {len(np.unique(g))} 颗")
    print(f"\n[托盘对照] 仅凭托盘光谱区分 Good/Bad 拍摄的 AUC = {tray_control(d):.3f}"
          "（0.5 = 无批次效应；越接近 1 越严重）")

    # 托盘归一化：把每个文件的托盘平均光谱缩放到两个文件的共同参考，逐波段s
    ref = d["tray_mean"].mean(axis=0)
    Xn = X * (ref / d["tray_mean"][d["source"]])

    rows = []
    for norm, XX in (("none", X), ("tray_scale", Xn)):
        for det_name in DETECTOR_NAMES:
            for prep in PREPS:
                _ids, lab, sc = berry_scores(XX, y, g, make_detector(det_name, prep))
                lo, hi = bootstrap_auc(lab, sc)
                rows.append({"norm": norm, "detector": det_name, "prep": prep,
                             "auc": roc_auc_score(lab, sc), "auc_lo": lo, "auc_hi": hi,
                             f"tpr@{int(FPR * 100)}%fpr": tpr_at_fpr(lab, sc)})
                r = rows[-1]
                print(f"{norm:10s} {det_name:12s} {prep:8s} AUC {r['auc']:.3f} "
                      f"[{lo:.3f}, {hi:.3f}]  检出率@5%误报 {r[f'tpr@{int(FPR * 100)}%fpr']:.3f}")
    df = pd.DataFrame(rows)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    df.to_csv("docs/results/blueberry_detection.csv", index=False)
    print("\n已保存 docs/results/blueberry_detection.csv")


if __name__ == "__main__":
    main()

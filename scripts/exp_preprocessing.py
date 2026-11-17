"""
步骤 2 核心实验：预处理 × 模型 × 划分方式，量化"预处理的价值"和"数据泄漏的危害"。

分类任务：预测类别（新鲜/陈旧/异物），指标 macro-F1
回归任务：只用食品像素预测水分，指标 R²
模型：简单模型（logreg / ridge）和高容量模型（kNN，容易"记住"物体）
划分：random（有泄漏） vs grouped（正确）

运行：uv run python scripts/exp_preprocessing.py
输出：docs/results/step2_preprocessing.csv 与 docs/figures/step2_leakage.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsi.data.simulator import SimConfig, generate_dataset
from hsi.preprocessing import MSC, SNV, SavitzkyGolay
from hsi.validation import build_pixel_table, evaluate

# 预处理方案：名称 -> 工厂函数（每次新建实例，避免状态共享）
PREPS = {
    "raw": lambda: "passthrough",
    "snv": lambda: SNV(),
    "msc": lambda: MSC(),
    "sg1": lambda: SavitzkyGolay(11, 2, deriv=1),
    "snv+sg1": lambda: make_pipeline(SNV(), SavitzkyGolay(11, 2, deriv=1)),
}

MODELS = {
    ("classification", "logreg"): lambda: LogisticRegression(max_iter=2000),
    ("classification", "knn"): lambda: KNeighborsClassifier(n_neighbors=5),
    ("regression", "ridge"): lambda: Ridge(alpha=1.0),
    ("regression", "knn"): lambda: KNeighborsRegressor(n_neighbors=5),
}
SCORING = {"classification": "f1_macro", "regression": "r2"}


def main() -> None:
    scenes = generate_dataset(24, SimConfig())
    X, y, moisture, groups = build_pixel_table(scenes, per_item=100)
    print(f"像素总数 {len(X)}，物体数 {len(np.unique(groups))}")

    food = np.isin(y, (1, 2))  # 回归只用食品像素（异物无水分）
    data = {
        "classification": (X, y, groups),
        "regression": (X[food], moisture[food], groups[food]),
    }

    rows = []
    for (task, model_name), model_fn in MODELS.items():
        Xt, yt, gt = data[task]
        for prep_name, prep_fn in PREPS.items():
            for scheme in ("random", "grouped"):
                pipe = make_pipeline(prep_fn(), StandardScaler(), model_fn())
                scores = evaluate(pipe, Xt, yt, gt, scheme, task, SCORING[task])
                rows.append(dict(task=task, model=model_name, prep=prep_name,
                                 scheme=scheme, mean=scores.mean(), std=scores.std()))
                print(f"{task:14s} {model_name:7s} {prep_name:8s} {scheme:8s} "
                      f"{scores.mean():.3f} ± {scores.std():.3f}")

    df = pd.DataFrame(rows)
    table = df.pivot_table(index=["task", "model", "prep"], columns="scheme", values="mean")
    table["gap(random-grouped)"] = table["random"] - table["grouped"]
    print("\n" + table.to_string(float_format=lambda v: f"{v:.3f}"))

    Path("docs/results").mkdir(parents=True, exist_ok=True)
    table.to_csv("docs/results/step2_preprocessing.csv")

    # ---- 作图：每个 (任务, 模型) 一个子图，对比两种划分 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    prep_order = list(PREPS)
    for ax, ((task, model), sub) in zip(axes.ravel(), df.groupby(["task", "model"])):
        p = sub.pivot(index="prep", columns="scheme", values="mean").reindex(prep_order)
        p[["random", "grouped"]].plot.bar(ax=ax, rot=0)
        ax.set_title(f"{task} / {model}  ({SCORING[task]})")
        ax.set_xlabel("")
    fig.tight_layout()
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("docs/figures/step2_leakage.png", dpi=130)
    print("已保存 docs/results/step2_preprocessing.csv 和 docs/figures/step2_leakage.png")


if __name__ == "__main__":
    main()

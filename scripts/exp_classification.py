"""
核心实验：5 种模型 x 4 种预处理，在"困难模式"数据上做分组交叉验证。

报告指标：
- pixel macro-F1 : 像素级，均值 ± 折间标准差
- object acc     : 物体级（同一物体像素多数票）准确率

运行：
  uv run python scripts/exp_classification.py --quick   # 约几分钟：只跑 sg1+snv，CNN 10 轮
  uv run python scripts/exp_classification.py           # 完整网格，约 10~20 分钟
输出：
  docs/results/step3_classification.csv
  docs/figures/step3_model_comparison.png
  docs/figures/step3_confusion_best.png
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from hsi.data.simulator import CLASS_NAMES, HARD_CONFIG, generate_dataset
from hsi.models.baselines import MODEL_NAMES, PREP_NAMES, make_baseline
from hsi.validation import build_pixel_table, object_level_accuracy, oof_predict

CNN_PREPS = ("raw", "sg1+snv")  # CNN 训练慢，只比较"不预处理"和"有预处理"两种


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速模式")
    args = ap.parse_args()

    preps = ("sg1+snv",) if args.quick else PREP_NAMES
    epochs = 10 if args.quick else 30

    scenes = generate_dataset(24, HARD_CONFIG)
    X, y, _, groups = build_pixel_table(scenes, per_item=60)
    print(f"像素 {len(X)}，物体 {len(np.unique(groups))}，类别分布 {np.bincount(y)[1:]}")

    rows, preds = [], {}
    for model in MODEL_NAMES:
        for prep in preps:
            if model == "cnn" and prep not in CNN_PREPS and not args.quick:
                continue
            pipe = make_baseline(model, prep, epochs=epochs)
            t0 = time.time()
            pred, splits = oof_predict(pipe, X, y, groups, n_jobs=1)
            fold_f1 = [f1_score(y[te], pred[te], average="macro") for _, te in splits]
            obj = object_level_accuracy(groups, y, pred)
            rows.append({"model": model, "prep": prep, "f1_mean": np.mean(fold_f1),
                         "f1_std": np.std(fold_f1), "object_acc": obj,
                         "seconds": time.time() - t0})
            preds[(model, prep)] = pred
            print(f"{model:7s} {prep:8s} F1 {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}  "
                  f"object acc {obj:.3f}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    df.to_csv("docs/results/step3_classification.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 图 1：模型 x 预处理 的 F1 对比
    ax = df.pivot(index="model", columns="prep", values="f1_mean").plot.bar(
        figsize=(9, 5), rot=0)
    ax.set_ylabel("pixel macro-F1 (grouped CV)")
    ax.legend(title="prep", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0.7, 1.0)    
    ax.set_title("Model x preprocessing on HARD simulated data")
    plt.tight_layout()
    plt.savefig("docs/figures/step3_model_comparison.png", dpi=130)

    # 图 2：最佳组合的归一化混淆矩阵（看错误集中在哪两类之间）
    best = df.sort_values("f1_mean", ascending=False).iloc[0]
    key = (best["model"], best["prep"])
    _fig, ax2 = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, preds[key], display_labels=CLASS_NAMES[1:], normalize="true", ax=ax2)
    ax2.set_title(f"Best: {key[0]} + {key[1]}")
    plt.tight_layout()
    plt.savefig("docs/figures/step3_confusion_best.png", dpi=130)
    print("已保存结果表和图")


if __name__ == "__main__":
    main()

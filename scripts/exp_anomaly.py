"""
步骤 4B 实验：只用正常食品训练，检测异物。
对比：3 种检测器 x 3 种预处理 x 2 种数据难度（普通异物 / 难检异物）。
指标：ROC-AUC、1% 误报率下的检出率。

运行：uv run python scripts/exp_anomaly.py
输出：docs/results/step4b_anomaly.csv, docs/figures/step4b_anomaly.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from hsi.data.simulator import FOREIGN, HARD_CONFIG, SUBTLE_CONFIG, generate_dataset
from hsi.models.anomaly import DETECTOR_NAMES, detection_rate_at_fpr, make_detector
from hsi.validation import build_pixel_table, oof_anomaly_scores

PREPS = ("raw", "snv", "sg1+snv")
DATASETS = {"obvious": HARD_CONFIG, "subtle": SUBTLE_CONFIG}


def main() -> None:
    rows = []
    for ds_name, cfg in DATASETS.items():
        scenes = generate_dataset(24, cfg)
        X, y, _, groups = build_pixel_table(scenes, per_item=60)
        is_foreign = y == FOREIGN
        print(f"[{ds_name}] 像素 {len(X)}，异物像素 {is_foreign.sum()}")
        for det_name in DETECTOR_NAMES:
            for prep in PREPS:
                scores = oof_anomaly_scores(make_detector(det_name, prep), X, y, groups)
                auc = roc_auc_score(is_foreign, scores)
                dr = detection_rate_at_fpr(is_foreign, scores, 0.01)
                rows.append({"data": ds_name, "detector": det_name, "prep": prep,
                             "auc": auc, "detect@1%fpr": dr})
                print(f"  {det_name:12s} {prep:8s} AUC {auc:.3f}  检出率@1%误报 {dr:.3f}")

    df = pd.DataFrame(rows)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    df.to_csv("docs/results/step4b_anomaly.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ds in zip(axes, DATASETS):
        p = df[df.data == ds].pivot(index="detector", columns="prep", values="detect@1%fpr")
        p.plot.bar(ax=ax, rot=0)
        ax.set_title(f"{ds} foreign: detection rate @ 1% FPR")
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("docs/figures/step4b_anomaly.png", dpi=130)
    print("已保存")


if __name__ == "__main__":
    main()

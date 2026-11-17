"""
步骤 4A 实验：水分回归，重点回答"R² 0.96 有多少来自分出新鲜/陈旧"。

同时报告：
- overall R² / RMSE : 新鲜+陈旧混合
- within-class R² / RMSE : 只看新鲜、只看陈旧（真正的水分预测能力）
- baseline : 只预测"类别均值"能得到多少 R²（这就是类间差异的贡献）

运行：uv run python scripts/exp_regression.py
输出：docs/results/step4a_regression.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from hsi.data.simulator import AGED, FRESH, HARD_CONFIG, generate_dataset
from hsi.models.regressors import REG_NAMES, make_reg_baseline
from hsi.validation import build_pixel_table, oof_regress

PREPS = ("raw", "snv", "sg1+snv")


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def main() -> None:
    X, y, moisture, groups = build_pixel_table(generate_dataset(24, HARD_CONFIG), per_item=60)
    food = np.isin(y, (FRESH, AGED))
    X, cls, m, g = X[food], y[food], moisture[food], groups[food]
    print(f"食品像素 {len(X)}")

    # 基线：只知道类别、预测类别均值，能解释多少方差
    cls_mean = np.where(cls == FRESH, m[cls == FRESH].mean(), m[cls == AGED].mean())
    print(f"【类别均值基线】overall R² = {r2_score(m, cls_mean):.3f} "
          f"（这部分是类间差异，不是真正的水分预测）")

    rows = []
    for model in REG_NAMES:
        for prep in PREPS:
            pred, _ = oof_regress(make_reg_baseline(model, prep), X, m, g)
            row = {"model": model, "prep": prep,
                   "R2_overall": r2_score(m, pred), "RMSE_overall": rmse(m, pred)}
            for name, c in (("fresh", FRESH), ("aged", AGED)):
                k = cls == c
                row[f"R2_{name}"] = r2_score(m[k], pred[k])
                row[f"RMSE_{name}"] = rmse(m[k], pred[k])
            rows.append(row)
            print(f"{model:6s} {prep:8s} overall R² {row['R2_overall']:.3f} | "
                  f"fresh R² {row['R2_fresh']:.3f} aged R² {row['R2_aged']:.3f} | "
                  f"RMSE fresh {row['RMSE_fresh']:.3f}")

    df = pd.DataFrame(rows)
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    df.to_csv("docs/results/step4a_regression.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

"""
DVC 阶段 2：评估三个任务，写 metrics/metrics.json，并记录到 MLflow。
运行：uv run python scripts/train.py [--no-mlflow] [--run-name 名字]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from hsi.evaluation import evaluate_all
from hsi.tracking import log_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mlflow", action="store_true", help="只写 metrics.json，不记录 MLflow")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    d = np.load("data/pixels.npz")
    metrics = evaluate_all(d["X"], d["y"], d["moisture"], d["groups"], params)

    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/metrics.json").write_text(json.dumps(metrics, indent=2))
    for k, v in metrics.items():
        print(f"{k:26s} {v:.4f}")

    if not args.no_mlflow:
        run_id = log_run(params, metrics,
                         artifacts=[Path("params.yaml"), Path("metrics/metrics.json")],
                         run_name=args.run_name)
        print(f"MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()

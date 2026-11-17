"""
把当前 metrics/metrics.json 设为基线 baselines/baseline_metrics.json。
只有确认新指标可接受时才运行，并且把基线文件的改动
作为代码审查的一部分提交。
运行：uv run python scripts/update_baseline.py
"""

import json
from pathlib import Path


def main() -> None:
    metrics = json.loads(Path("metrics/metrics.json").read_text())
    Path("baselines").mkdir(exist_ok=True)
    Path("baselines/baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("已更新 baselines/baseline_metrics.json：")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

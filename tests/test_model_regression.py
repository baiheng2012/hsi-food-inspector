"""
模型回归测试：用 params.yaml 的配置从头评估，指标不得低于已提交的基线。
这条测试进入 CI 后，任何让指标变差的改动都无法合并。
运行：uv run pytest -m model -v
"""

import json
from pathlib import Path

import pytest
import yaml

from hsi.evaluation import run_evaluation
from hsi.regression_check import compare_to_baseline

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.model


def test_metrics_not_worse_than_baseline():
    baseline_path = ROOT / "baselines" / "baseline_metrics.json"
    if not baseline_path.exists():
        pytest.fail("缺少基线：先运行 dvc repro，再运行 scripts/update_baseline.py 并提交")
    baseline = json.loads(baseline_path.read_text())
    params = yaml.safe_load((ROOT / "params.yaml").read_text())

    current = run_evaluation(params)

    failures = compare_to_baseline(current, baseline)
    assert not failures, "模型指标退化：\n" + "\n".join(failures)

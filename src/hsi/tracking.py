"""
MLflow 封装 (tracking.py)
==========================

把"记录一次实验"收敛成一个函数 log_run，业务代码不直接依赖 MLflow 的细节。
后端用 SQLite 文件（mlflow.db）：不需要起服务器，且各 MLflow 版本都支持。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow

DEFAULT_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "hsi-food-inspector"


def flatten(d: dict, prefix: str = "") -> dict:
    """嵌套字典展平：{"a": {"b": 1}} -> {"a.b": 1}。MLflow 的参数必须是扁平的。"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _git_commit() -> str:
    """当前 git 提交号，记入实验标签，方便从实验反查代码版本。取不到就返回 unknown。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"    
    


def log_run(params: dict, metrics: dict, artifacts: list[Path] | None = None,
            run_name: str | None = None, experiment: str = DEFAULT_EXPERIMENT,
            tracking_uri: str = DEFAULT_URI) -> str:
    """
    记录一次实验，返回 run_id。

    params    : 可嵌套字典，会被展平后记录
    metrics   : {指标名: 数值}
    artifacts : 要一并保存的文件（如 params.yaml、metrics.json）
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.log_params(flatten(params))
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id

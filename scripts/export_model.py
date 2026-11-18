"""
导出部署用引擎：按 params.yaml 训练三个模型，融合成 float32 引擎，存为 models/engine.npz。
Docker 构建时会运行它，所以镜像里自带模型。

运行：uv run python scripts/export_model.py [--out models/engine.npz]
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from hsi.streaming.engine import FusedEngine, fit_models
from hsi.streaming.engine_opt import OptimizedEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/engine.npz")
    args = ap.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    base = FusedEngine.from_sklearn(fit_models(params))  # 只支持 raw 预处理，否则明确报错
    engine = OptimizedEngine(base, np.float32)
    engine.save(args.out)
    size_kb = Path(args.out).stat().st_size / 1024
    print(f"已导出 {args.out}（{size_kb:.0f} KB），波段数 {engine.n_bands}，"
          f"类别 {engine.classes.tolist()}")


if __name__ == "__main__":
    main()

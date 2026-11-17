"""
DVC 阶段 1：生成像素表，保存为 data/pixels.npz。
运行：uv run python scripts/prepare_data.py   （通常由 dvc repro 调用）
"""

from pathlib import Path

import numpy as np
import yaml

from hsi.evaluation import build_data


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    X, y, moisture, groups = build_data(params["data"])
    Path("data").mkdir(exist_ok=True)
    # float32 足够，节省空间；groups 是物体 ID，后面分组交叉验证要用
    np.savez_compressed("data/pixels.npz", X=X.astype(np.float32), y=y,
                        moisture=moisture.astype(np.float32), groups=groups)
    print(f"已保存 data/pixels.npz：像素 {len(X)}，物体 {len(np.unique(groups))}")


if __name__ == "__main__":
    main()

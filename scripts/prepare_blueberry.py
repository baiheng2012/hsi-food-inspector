"""
把 data/blueberry/raw 下的 ENVI 文件处理成 data/blueberry/processed/blueberry.npz。
运行：uv run python scripts/prepare_blueberry.py [--per-item 200]
"""

import argparse
from pathlib import Path

import numpy as np

from hsi.data.blueberry import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/blueberry/raw")
    ap.add_argument("--out", default="data/blueberry/processed/blueberry.npz")
    ap.add_argument("--per-item", type=int, default=200)
    args = ap.parse_args()

    d = build_dataset(Path(args.raw), per_item=args.per_item)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **d)
    n_b = len(np.unique(d["groups"]))
    print(f"\n已保存 {args.out}：像素 {len(d['X'])}，波段 {d['X'].shape[1]}"
          f"（{d['wavelengths'][0]:.0f}~{d['wavelengths'][-1]:.0f} nm），果 {n_b} 颗，"
          f"好果像素 {(d['y'] == 0).sum()}，缺陷果像素 {(d['y'] == 1).sum()}，"
          f"托盘像素 {len(d['X_tray'])}")


if __name__ == "__main__":
    main()

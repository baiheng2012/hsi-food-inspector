"""
线扫描模拟 (linescan.py)
=========================

pushbroom 相机每个时刻只输出传送带上的一行：形状 (W, B)，W 个像素，每像素 B 个波段。
把整个立方体 (H, W, B) 沿高度方向逐行取出，就是一条"数据流"。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iter_lines(cube: np.ndarray) -> Iterator[np.ndarray]:
    """
    按行产生数据：第 i 次产生 cube[i]，形状 (W, B)。
    这是生成器：不预先复制数据，用到哪一行才取哪一行，行为上等价于真实相机的数据流。
    """
    if cube.ndim != 3:
        raise ValueError(f"期望立方体形状 (H, W, B)，实际 {cube.shape}")
    for i in range(cube.shape[0]):
        yield cube[i]

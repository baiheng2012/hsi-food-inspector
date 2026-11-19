"""
蓝莓高光谱数据加载 (blueberry.py)
=================================

流程：读取 ENVI(.bil) -> 用白参考校正成反射率 -> 在 800 nm 上分割出每颗蓝莓
-> 裁剪并合并波段 -> 输出像素表。

反射率 = DN / 白参考(逐像素位置) * (白参考快门 / 样本快门)
暗电流：实测黑带 DN 约 1~2（白板 1500~2500），忽略不计。

输出 npz 字段
-------------
X, y(0 好果/1 缺陷果), groups(每颗果唯一 ID), source(文件序号), wavelengths,
X_tray, tray_source, tray_row (托盘像素，用作批次效应的阴性对照),
tray_mean (每个文件的托盘平均光谱)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spectral
from scipy import ndimage as ndi

STEP = 4            # 空间下采样步长（行、列都隔 4 取 1）
WL_MIN, WL_MAX = 450.0, 950.0
BIN = 4             # 相邻 4 个波段取平均


def open_cube(bil: Path):
    """返回 (memmap 立方体 (lines, samples, bands), 波长数组, 快门)。"""
    img = spectral.envi.open(str(bil) + ".hdr", str(bil))
    return img.open_memmap(), np.array(img.bands.centers), float(str(img.metadata["shutter"]))


def white_profile(cube, chunk: int = 100) -> np.ndarray:
    """白参考沿行方向平均 -> (samples, bands)。分块读取，避免一次载入 2.7 GB。"""
    acc = np.zeros(cube.shape[1:], dtype=np.float64)
    for i in range(0, cube.shape[0], chunk):
        acc += np.asarray(cube[i:i + chunk]).sum(axis=0, dtype=np.float64)
    return (acc / cube.shape[0]).astype(np.float32)

"""
def otsu(x: np.ndarray, bins: int = 256) -> float:
    
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    cum = np.cumsum(p * c)
    m0 = cum / np.maximum(w0, 1e-12)
    m1 = (cum[-1] - cum) / np.maximum(w1, 1e-12)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])
"""

def otsu(x: np.ndarray, bins: int = 256) -> float:
    """
    大津法阈值：使两类的类间方差最大。
    两个峰之间若有空隙，空隙内所有阈值的类间方差并列最大，此时取并列区间的中点，
    使阈值落在空隙正中间，而不是贴着某一个峰的边缘。
    """
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    cum = np.cumsum(p * c)
    m0 = cum / np.maximum(w0, 1e-12)
    m1 = (cum[-1] - cum) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    ties = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(c[ties[len(ties) // 2]])

def bin_bands(X: np.ndarray, wl: np.ndarray, wl_min=WL_MIN, wl_max=WL_MAX, k=BIN):
    """裁剪到 [wl_min, wl_max]，再把相邻 k 个波段取平均。返回 (X', wl')。"""
    idx = np.flatnonzero((wl >= wl_min) & (wl <= wl_max))
    idx = idx[: len(idx) // k * k]  # 去掉不足 k 个的尾巴
    xb = X[..., idx].reshape(*X.shape[:-1], len(idx) // k, k).mean(axis=-1)
    return xb.astype(np.float32), wl[idx].reshape(-1, k).mean(axis=1).astype(np.float32)


def segment_berries(r800: np.ndarray, valid: np.ndarray, margin: float = 0.04,
                    min_area: int = 150, ratio=(0.6, 1.6)):
    """
    在 800 nm 反射率图上分割蓝莓（滞后阈值）。
    高阈值(大津)只用来确认"这里有果实"，低阈值(托盘中位数+margin)决定范围，
    这样较暗的缺陷区域不会被切掉。再按面积、长宽比、是否触边筛选。
    返回 (标签图 0=非果, 1..n=第 k 颗果, 低阈值)。
    """
    high = otsu(r800[valid])
    low = float(np.median(r800[valid])) + margin
    cand = ndi.binary_opening(valid & (r800 > low), iterations=2)
    lab0, _ = ndi.label(cand)
    seeds = np.unique(lab0[valid & (r800 > high)])
    mask = ndi.binary_fill_holes(np.isin(lab0, seeds[seeds > 0]))
    lab, _ = ndi.label(mask)
    out = np.zeros_like(lab)
    kept = 0
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == k).sum())
        touches = (sl[0].start == 0 or sl[1].start == 0
                   or sl[0].stop == lab.shape[0] or sl[1].stop == lab.shape[1])
        if area >= min_area and ratio[0] <= h / w <= ratio[1] and not touches:
            kept += 1
            out[sl][lab[sl] == k] = kept
    return out, low


def load_scene(bil: Path, white: np.ndarray, white_shutter: float, per_item: int,
               rng: np.random.Generator, n_tray: int = 3000) -> dict:
    """处理一个文件：返回每颗果的像素、托盘像素、托盘平均光谱。"""
    cube, wl, sh = open_cube(bil)
    small = np.asarray(cube[::STEP, ::STEP, :], dtype=np.float32)  # (450, 400, 462)
    refl = small / np.maximum(white[::STEP], 1.0) * (white_shutter / sh)

    b800 = int(np.argmin(np.abs(wl - 800)))
    r800 = refl[:, :, b800]
    col_ok = white[::STEP, b800] > 0.2 * np.median(white[:, b800])  # 白板信号太弱的列不可信
    valid = np.broadcast_to(col_ok[None, :], r800.shape)
    lab, low = segment_berries(r800, valid)

    inner = np.zeros_like(lab)  # 向内腐蚀 1 个缩略图像素，去掉与托盘/阴影混合的边缘
    for k in range(1, lab.max() + 1):
        inner[ndi.binary_erosion(lab == k)] = k

    berries = []
    for k in range(1, lab.max() + 1):
        pts = np.argwhere(inner == k)
        if len(pts) > per_item:
            pts = pts[rng.choice(len(pts), per_item, replace=False)]
        berries.append(bin_bands(refl[pts[:, 0], pts[:, 1]], wl)[0])

    tray_pts = np.argwhere(valid & ~ndi.binary_dilation(lab > 0, iterations=5) & (r800 < low))
    tray_pts = tray_pts[rng.choice(len(tray_pts), min(n_tray, len(tray_pts)), replace=False)]
    xt, wlb = bin_bands(refl[tray_pts[:, 0], tray_pts[:, 1]], wl)
    return {"berries": berries, "tray": xt, "tray_row": tray_pts[:, 0], "wl": wlb}


def build_dataset(raw_dir: Path, white_name: str = "WhiteReference.bil",
                  per_item: int = 200, seed: int = 0) -> dict:
    """处理 raw_dir 下所有 'Good Blueberry*.bil' 和 'Bad Blueberry*.bil'。"""
    rng = np.random.default_rng(seed)
    wcube, _, wsh = open_cube(raw_dir / white_name)
    print("计算白参考平均光谱 ...")
    white = white_profile(wcube)

    files = [(p, 0) for p in sorted(raw_dir.glob("Good Blueberry*.bil"))] + \
            [(p, 1) for p in sorted(raw_dir.glob("Bad Blueberry*.bil"))]
    if not files:
        raise FileNotFoundError(f"{raw_dir} 下没有找到 Good/Bad Blueberry*.bil")

    X, y, g, src, Xt, ts, tr, tmean, wl = [], [], [], [], [], [], [], [], None
    for fi, (path, lab) in enumerate(files):
        print(f"处理 {path.name} ...")
        s = load_scene(path, white, wsh, per_item, rng)
        wl = s["wl"]
        for k, xb in enumerate(s["berries"], start=1):
            X.append(xb)
            y.append(np.full(len(xb), lab))
            g.append(np.full(len(xb), fi * 1000 + k))   # 每颗果一个唯一 ID，跨文件不重复
            src.append(np.full(len(xb), fi))
        Xt.append(s["tray"])
        ts.append(np.full(len(s["tray"]), fi))
        tr.append(s["tray_row"])
        tmean.append(s["tray"].mean(axis=0))
        print(f"  果 {len(s['berries'])} 颗，像素 {sum(len(b) for b in s['berries'])}")
    return {
        "X": np.concatenate(X), "y": np.concatenate(y), "groups": np.concatenate(g),
        "source": np.concatenate(src), "wavelengths": wl,
        "X_tray": np.concatenate(Xt), "tray_source": np.concatenate(ts),
        "tray_row": np.concatenate(tr), "tray_mean": np.stack(tmean),
    }

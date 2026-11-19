"""
查看一个 ENVI(.bil) 文件：形状、数值范围、是否饱和，并保存缩略图。
用内存映射读取，不会把 2.66 GB 全部载入内存。

运行：uv run python scripts/inspect_blueberry.py "路径/Bad Blueberry 1-42.bil.hdr"
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral

arg = Path(sys.argv[1])
# 传 xxx.bil 或 xxx.bil.hdr 都可以
bil = arg.with_suffix("") if arg.suffix == ".hdr" else arg
hdr = Path(str(bil) + ".hdr")
img = spectral.envi.open(str(hdr), str(bil))

cube = img.open_memmap()  # 形状 (lines, samples, bands)，按需读取
wl = np.array(img.bands.centers)
print("形状 (行, 像素, 波段):", cube.shape, "dtype:", cube.dtype)
print(f"波长 {wl[0]:.1f} ~ {wl[-1]:.1f} nm，共 {len(wl)} 个")


def band(nm: float) -> int:
    """最接近某波长的波段序号。"""
    return int(np.argmin(np.abs(wl - nm)))


# 每 4 行、4 列取一个像素做缩略图，只读需要的 3 个波段
b_r, b_g, b_b = band(640), band(550), band(460)
small = np.asarray(cube[::4, ::4, [b_r, b_g, b_b]], dtype=np.float32)
print("缩略图形状:", small.shape)

lo, hi = np.percentile(small, 1), np.percentile(small, 99)
print(f"数值分位数 1%={lo:.0f}  99%={hi:.0f}  最大={small.max():.0f}  最小={small.min():.0f}")
print(f"等于上限 8190 的比例: {(small >= 8190).mean() * 100:.3f}%")

rgb = np.clip((small - lo) / (hi - lo + 1e-9), 0, 1)
nir = np.asarray(cube[::4, ::4, band(800)], dtype=np.float32)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(rgb)
ax[0].set_title("pseudo-RGB (640/550/460 nm)")
ax[1].imshow(nir, cmap="gray")
ax[1].set_title("800 nm")
# 每隔 20 个像素取点，画几条平均光谱，看形状和数量级
ax[2].plot(wl, np.asarray(cube[::40, ::40, :], dtype=np.float32).reshape(-1, len(wl)).mean(0))
ax[2].set_title("mean spectrum of sampled pixels")
ax[2].set_xlabel("nm")
out = Path("docs/figures")
out.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out / "blueberry_inspect.png", dpi=110)
print("已保存", out / "blueberry_inspect.png")

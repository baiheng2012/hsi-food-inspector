"""
蓝莓数据校正与分割检查。

反射率 = DN / 白参考 * (白参考快门 / 样本快门)
白参考沿行方向取平均，得到 (samples, bands)，同时校正传送带宽度方向的光照不均。
为省内存，样本图像按 STEP 隔行隔列读取（只用于检查，不是最终加载器）。

运行：
uv run python scripts/calibrate_blueberry.py \
  "data/blueberry/raw/Bad Blueberry 1-42.bil" data/blueberry/raw/WhiteReference.bil
输出：docs/figures/blueberry_calibrated.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spectral
from scipy import ndimage as ndi

STEP = 4
CEILING = 8190


def open_cube(bil: Path):
    img = spectral.envi.open(str(bil) + ".hdr", str(bil))
    shutter = float(str(img.metadata["shutter"]))
    return img.open_memmap(), np.array(img.bands.centers), shutter


def white_profile(cube, chunk: int = 100):
    """沿行方向平均，返回 (samples, bands) 的白板平均 DN，以及饱和像素比例。分块读取避免爆内存。"""
    acc = np.zeros(cube.shape[1:], dtype=np.float64)
    sat = 0
    for i in range(0, cube.shape[0], chunk):
        block = np.asarray(cube[i:i + chunk])
        acc += block.sum(axis=0, dtype=np.float64)
        sat += int((block >= CEILING).sum())
    return (acc / cube.shape[0]).astype(np.float32), sat / cube.size


def otsu(x: np.ndarray, bins: int = 256) -> float:
    """大津法阈值：使前景/背景两类的类间方差最大。"""
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    cum = np.cumsum(p * c)
    m0 = cum / np.maximum(w0, 1e-12)
    m1 = (cum[-1] - cum) / np.maximum(w1, 1e-12)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def main() -> None:
    bil, white_bil = Path(sys.argv[1]), Path(sys.argv[2])
    cube, wl, sh = open_cube(bil)
    wcube, wwl, shw = open_cube(white_bil)
    assert cube.shape == wcube.shape and np.allclose(wl, wwl), "样本与白参考的形状/波长不一致"

    def band(nm: float) -> int:
        return int(np.argmin(np.abs(wl - nm)))

    print("计算白参考平均光谱（读取约 2.7 GB，请稍等）...")
    W, sat = white_profile(wcube)
    print(f"白参考 DN：最大 {W.max():.0f}，饱和(>=8190)比例 {sat * 100:.4f}%")
    for nm in (450, 550, 650, 800, 950):
        col = W[:, band(nm)]
        print(f"  {nm} nm：白板 DN 中位 {np.median(col):.0f}，最小 {col.min():.0f}")

    small = np.asarray(cube[::STEP, ::STEP, :], dtype=np.float32)
    print(f"缩略立方体 {small.shape}")
    # 暗电流粗查：图像左侧黑带（托盘外）的 DN，若明显大于 0，说明有暗偏置
    print("左侧黑带(前 40 列)平均 DN：",
          {nm: round(float(small[:, :40 // STEP, band(nm)].mean()), 1) for nm in (450, 650, 800, 950)})

    Ws = W[::STEP]                                       # (400, 462)
    refl = small / np.maximum(Ws, 1.0) * (shw / sh)
    b800 = band(800)
    valid = Ws[:, b800] > 0.2 * np.median(Ws[:, b800])   # 白板信号足够强的列才可信
    valid = np.broadcast_to(valid[None, :], refl.shape[:2])
    r800 = refl[:, :, b800]
    #thr = otsu(r800[valid])
    #mask = ndi.binary_fill_holes(ndi.binary_opening(valid & (r800 > thr)))
    #lab, n = ndi.label(mask)
    thr = otsu(r800[valid])                       # 高阈值：确认"这里有果实"
    tray = float(np.median(r800[valid]))          # 托盘占大部分面积，中位数近似托盘反射率
    low = tray + 0.04                             # 低阈值：包含较暗的缺陷区域
    cand = ndi.binary_opening(valid & (r800 > low), iterations=2)
    lab0, _ = ndi.label(cand)
    seeds = np.unique(lab0[valid & (r800 > thr)])
    mask = ndi.binary_fill_holes(np.isin(lab0, seeds[seeds > 0]))
    lab, n = ndi.label(mask)
    print(f"托盘 800 nm 反射率 {tray:.3f}，低阈值 {low:.3f}，高阈值 {thr:.3f}")   
    
    
    
    keep = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == k).sum())
        touches = sl[0].start == 0 or sl[1].start == 0 or \
            sl[0].stop == lab.shape[0] or sl[1].stop == lab.shape[1]
        #if area >= 150 and 0.6 <= h / w <= 1.6 and not touches:
        #    keep.append(k)
        if area >= 150 and 0.6 <= h / w <= 1.6 and not touches:
            keep.append(k)
        else:
            print(f"  剔除连通域 {k}：面积 {area}，高宽比 {h / w:.2f}，触边 {touches}，"
                  f"位置 y={sl[0].start}~{sl[0].stop} x={sl[1].start}~{sl[1].stop}")


    areas = np.array([(lab == k).sum() for k in keep])
    print(f"\n800 nm 反射率阈值 {thr:.3f}，连通域 {n} 个，筛选后 {len(keep)} 个"
          f"（期望 42），面积(缩略图像素) 最小 {areas.min()} 中位 {int(np.median(areas))} 最大 {areas.max()}")

    spectra = np.stack([refl[lab == k].mean(axis=0) for k in keep])
    board = refl[valid & ~ndi.binary_dilation(mask, iterations=5)].mean(axis=0)
    print(f"反射率>1.2 的像素比例：{(refl[valid] > 1.2).mean() * 100:.3f}%")
    for nm in (450, 550, 680, 800, 900):
        b = band(nm)
        print(f"  {nm} nm：蓝莓平均 {spectra[:, b].mean():.3f}（最小 {spectra[:, b].min():.3f} "
              f"最大 {spectra[:, b].max():.3f}），托盘 {board[b]:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(r800, cmap="gray", vmin=0, vmax=np.percentile(r800[valid], 99))
    ax[0].contour(np.isin(lab, keep), levels=[0.5], colors="r", linewidths=0.6)
    for i, k in enumerate(keep):
        cy, cx = ndi.center_of_mass(lab == k)
        ax[0].text(cx, cy, str(i), color="cyan", fontsize=7, ha="center")
    ax[0].set_title(f"800 nm reflectance + segmentation ({len(keep)} objects)")
    for s in spectra:
        ax[1].plot(wl, s, color="tab:blue", alpha=0.25, lw=0.8)
    ax[1].plot(wl, spectra.mean(axis=0), color="navy", lw=2, label="mean of berries")
    ax[1].plot(wl, board, color="tab:orange", lw=2, label="tray")
    ax[1].set_ylim(0, 1.2)
    ax[1].set_xlabel("nm")
    ax[1].set_ylabel("reflectance")
    ax[1].legend()
    ax[1].set_title("calibrated reflectance per berry")
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    #fig.savefig(out / "blueberry_calibrated.png", dpi=110)
    #print("已保存", out / "blueberry_calibrated.png")
    fig.savefig(out / f"blueberry_calibrated_{bil.stem.replace(' ', '_')}.png", dpi=110)
    print("已保存", out / f"blueberry_calibrated_{bil.stem.replace(' ', '_')}.png")

if __name__ == "__main__":
    main()

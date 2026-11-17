"""
可视化脚本：生成一个场景，保存三张子图到 docs/figures/scene_overview.png
  1. 伪彩色图（取三个波段当 RGB）
  2. 类别标签图
  3. 每个类别的平均光谱曲线（看看四类光谱到底有什么区别）

运行：python scripts/visualize_scene.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，WSL 下直接保存文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np

from hsi.data.simulator import CLASS_NAMES, SimConfig, generate_scene


def main() -> None:
    cfg = SimConfig()
    scene = generate_scene(cfg, scene_id=0)
    cube, labels, wl = scene.cube, scene.label_map, scene.wavelengths

    # 伪彩色：取 1/4、1/2、3/4 位置的三个波段，并各自拉伸到 [0,1] 方便观看
    b = cube.shape[2]
    rgb = np.stack([cube[:, :, b // 4], cube[:, :, b // 2], cube[:, :, 3 * b // 4]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].imshow(rgb)
    axes[0].set_title("Pseudo-RGB (3 bands)")
    axes[0].axis("off")

    im = axes[1].imshow(labels, cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title("Label map (0 bg, 1 fresh, 2 aged, 3 foreign)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3], fraction=0.046)

    # 每类平均光谱 ± 1 个标准差
    for c, name in enumerate(CLASS_NAMES):
        px = cube[labels == c]
        if len(px) == 0:
            continue
        mean, std = px.mean(axis=0), px.std(axis=0)
        axes[2].plot(wl, mean, label=name)
        axes[2].fill_between(wl, mean - std, mean + std, alpha=0.2)
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("Reflectance")
    axes[2].set_title("Mean spectra per class (±1 std)")
    axes[2].legend()

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "scene_overview.png", dpi=130)
    print(f"已保存: {out / 'scene_overview.png'}")


if __name__ == "__main__":
    main()

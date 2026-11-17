"""
可视化预处理效果：取"新鲜食品"像素，对比 原始 / SNV / MSC / SG一阶导 之后的光谱带。
预期：SNV、MSC 之后色带明显变窄（散射被抵消）。

运行：uv run python scripts/visualize_preprocessing.py
输出：docs/figures/preprocessing_effect.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsi.data.simulator import FRESH, SimConfig, generate_scene
from hsi.preprocessing import MSC, SNV, SavitzkyGolay


def main() -> None:
    scene = generate_scene(SimConfig(), 0)
    X_fg, y_fg, _, _ = scene.flatten(foreground_only=True)
    wl = scene.wavelengths

    # 仅用于可视化：MSC 的参考光谱取前景像素均值（正式实验中一定放进 Pipeline）
    variants = {
        "Raw": X_fg,
        "SNV": SNV().fit_transform(X_fg),
        "MSC": MSC().fit_transform(X_fg),
        "SG 1st derivative": SavitzkyGolay(11, 2, deriv=1).fit_transform(X_fg),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (name, data) in zip(axes.ravel(), variants.items()):
        fresh = data[y_fg == FRESH]
        mean, std = fresh.mean(axis=0), fresh.std(axis=0)
        ax.plot(wl, mean)
        ax.fill_between(wl, mean - std, mean + std, alpha=0.25)
        ax.set_title(f"Fresh pixels: {name}")
        ax.set_xlabel("Wavelength (nm)")

    out = Path("docs/figures"); out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / "preprocessing_effect.png", dpi=130)
    print(f"已保存: {out / 'preprocessing_effect.png'}")


if __name__ == "__main__":
    main()

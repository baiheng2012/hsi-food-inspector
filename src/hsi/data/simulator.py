"""
高光谱场景模拟器 (simulator.py)
================================

本模块负责生成"传送带上的食品"高光谱图像，用于在没有真实数据时
验证整条机器学习流水线（预处理 -> 训练 -> 验证 -> 实时推理）。

核心概念
--------
- 立方体 (cube)：形状 (H, W, B) 的 float32 数组。每个像素是一条长度为 B 的光谱。
- 标签图 (label_map)：形状 (H, W) 的 int 数组，每个像素的类别。
- 水分图 (moisture_map)：形状 (H, W) 的 float 数组，食品像素有水分真值，其余为 NaN。
- 物体 ID 图 (item_map)：形状 (H, W)，每个物体一个全局唯一 ID，背景为 0。
  用于"按物体分组"的交叉验证，防止数据泄漏。

对外主要接口
------------
- SimConfig：模拟参数
- Scene：一个场景的全部数据（dataclass）
- generate_scene(cfg, scene_id)：生成单个场景
- generate_dataset(n_scenes, cfg)：生成多个场景
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BACKGROUND = 0  # 传送带
FRESH = 1       # 新鲜食品
AGED = 2        # 陈旧食品
FOREIGN = 3     # 异物（如塑料）
CLASS_NAMES = ["background", "fresh", "aged", "foreign"]


@dataclass(frozen=True)
class SimConfig:
    """模拟器参数。frozen=True 表示创建后不可修改，保证可复现。"""

    height: int = 128            # 图像高度（沿传送带方向的行数）
    width: int = 128             # 图像宽度（垂直于传送带方向，传感器线的像素数）
    n_bands: int = 100           # 波段数
    wl_min: float = 900.0        # 起始波长 (nm)
    wl_max: float = 1700.0       # 结束波长 (nm)
    n_items: int = 8             # 每个场景放置的物体数量
    class_probs: tuple = (0.45, 0.35, 0.20)  # 物体为 新鲜/陈旧/异物 的概率
    noise_std: float = 0.01      # 加性高斯噪声标准差（传感器噪声）
    scatter_std: float = 0.15    # 乘性散射标准差（光照/表面不均，相对于 1.0）
    offset_std: float = 0.02     # 加性基线偏移标准差
    item_var_std: float = 0.02   # 物体之间的个体光谱差异
    slope_std: float = 0.0           # 线性斜率基线漂移的标准差（0 = 关闭）
    subtle_foreign: float = 0.0      # 0=异物光谱与原来相同；越接近 1，异物光谱越像食品（越难检）    
    aged_moisture_mean: float = 0.50 # 陈旧食品的平均水分（越接近新鲜的 0.80，越难区分）
    aged_browning: float = 0.12      # 陈旧食品的褐变强度（越小，与新鲜越难区分）    
    seed: int = 0                # 基础随机种子


@dataclass
class Scene:
    """一个场景（一次传送带扫描）的全部数据。"""

    cube: np.ndarray            # (H, W, B) float32 反射率光谱
    label_map: np.ndarray       # (H, W) int8 类别
    moisture_map: np.ndarray    # (H, W) float32 水分，非食品为 NaN
    item_map: np.ndarray        # (H, W) int32 物体全局唯一 ID，背景为 0
    wavelengths: np.ndarray     # (B,) float32 波长 (nm)
    scene_id: int = 0
    meta: dict = field(default_factory=dict)

    def flatten(self, foreground_only: bool = False):
        """
        把立方体展开为"像素表"，供 sklearn 等按样本训练的模型使用。

        参数
        ----
        foreground_only : 为 True 时丢弃背景像素（只保留食品和异物）。

        返回
        ----
        X        : (N, B) 光谱矩阵
        y        : (N,)   类别标签
        moisture : (N,)   水分（非食品为 NaN）
        groups   : (N,)   物体 ID，用于 GroupKFold
        """
        _h, _w, b = self.cube.shape
        x = self.cube.reshape(-1, b)
        y = self.label_map.reshape(-1)
        m = self.moisture_map.reshape(-1)
        g = self.item_map.reshape(-1)
        if foreground_only:
            keep = y != BACKGROUND
            x, y, m, g = x[keep], y[keep], m[keep], g[keep]
        return x, y, m, g

# 困难模式：更强的散射 + 斜率漂移 + 个体差异大 + 新鲜/陈旧更接近。
# 默认 SimConfig() 完全不受影响，步骤 1~2 的结果可复现。
HARD_CONFIG = SimConfig(
    scatter_std=0.25,
    slope_std=0.10,
    item_var_std=0.05,
    aged_moisture_mean=0.72,
    aged_browning=0.05,
)

# 异物检测专用：在困难模式基础上，异物光谱 75% 接近新鲜食品
SUBTLE_CONFIG = SimConfig(
    scatter_std=0.25, slope_std=0.10, item_var_std=0.05,
    aged_moisture_mean=0.72, aged_browning=0.05,
    subtle_foreign=0.75,
)

def make_wavelengths(cfg: SimConfig) -> np.ndarray:
    """生成等间距的波长数组，形状 (B,)，单位 nm。"""
    return np.linspace(cfg.wl_min, cfg.wl_max, cfg.n_bands, dtype=np.float32)


def _gauss(wl: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    以 center 为中心、width 为标准差的高斯曲线，用来模拟吸收峰。
    返回形状与 wl 相同，峰值为 1。
    """
    return np.exp(-0.5 * ((wl - center) / width) ** 2)


#def base_spectrum(wl: np.ndarray, cls: int, moisture: float = 0.0) -> np.ndarray:
def base_spectrum(wl: np.ndarray, cls: int, moisture: float = 0.0,
                  browning: float = 0.12) -> np.ndarray:
    """
    生成某类别的"理想光谱"（无噪声、无散射）。

    物理直觉（简化）：
    - 水在 ~970、~1200、~1450 nm 有强吸收，水分越高吸收越深 -> 反射率越低。
    - 陈旧食品：水分更低，并且在短波端有褐变引起的额外吸收。
    - 异物（塑料）：整体更亮，在 ~1210 和 ~1650 nm 有 C-H 键吸收峰，
      形状与水的吸收明显不同。
    - 背景（传送带）：很暗且接近平坦。
    """
    x = (wl - wl.min()) / (wl.max() - wl.min())  # 归一化到 [0, 1]

    if cls == BACKGROUND:
        spec = 0.05 + 0.03 * x
    elif cls in (FRESH, AGED):
        baseline = 0.55 + 0.15 * x  # 平滑的基础反射
        water = moisture * (
            0.25 * _gauss(wl, 970, 30)
            + 0.35 * _gauss(wl, 1200, 40)
            + 0.50 * _gauss(wl, 1450, 50)
        )
        sugar = 0.04 * _gauss(wl, 910, 20)  # 一个很弱的"糖"特征
        spec = baseline - water - sugar
        if cls == AGED:
            # 褐变：短波端指数衰减形式的额外吸收
            #spec = spec - 0.12 * np.exp(-x / 0.15)
            spec = spec - browning * np.exp(-x / 0.15)            
    elif cls == FOREIGN:
        baseline = 0.75 - 0.10 * x
        spec = baseline - 0.15 * _gauss(wl, 1210, 15) - 0.20 * _gauss(wl, 1650, 20)
    else:
        raise ValueError(f"未知类别: {cls}")

    return np.clip(spec, 0.01, 1.0).astype(np.float32)


#def _sample_moisture(rng: np.random.Generator, cls: int) -> float:
    """按类别采样一个物体的水分真值（0~1 之间的相对量）。异物无水分，返回 NaN。"""
#    if cls == FRESH:
#        return float(np.clip(rng.normal(0.80, 0.05), 0.4, 1.0))
#    if cls == AGED:
#        return float(np.clip(rng.normal(0.50, 0.08), 0.1, 0.8))
#    return float("nan")
def _sample_moisture(rng: np.random.Generator, cls: int, cfg: SimConfig) -> float:
    """按类别采样一个物体的水分真值（0~1 之间的相对量）。异物无水分，返回 NaN。"""
    if cls == FRESH:
        return float(np.clip(rng.normal(0.80, 0.05), 0.4, 1.0))
    if cls == AGED:
        return float(np.clip(rng.normal(cfg.aged_moisture_mean, 0.08), 0.1, 0.8))
    return float("nan")

def generate_scene(cfg: SimConfig, scene_id: int = 0) -> Scene:
    """
    生成一个场景。

    流程
    ----
    1. 初始化为传送带背景。
    2. 随机放置 n_items 个椭圆形物体，每个物体随机类别、水分、个体光谱差异。
    3. 对每个像素施加真实感干扰：乘性散射、加性偏移、传感器噪声。

    参数
    ----
    cfg      : 模拟参数
    scene_id : 场景编号。相同 (cfg, scene_id) 一定产生相同结果（可复现）。
    """
    # 每个场景使用独立的随机数生成器，保证可复现且场景之间互不相同
    rng = np.random.default_rng(cfg.seed + scene_id)
    h, w, b = cfg.height, cfg.width, cfg.n_bands
    wl = make_wavelengths(cfg)

    label_map = np.zeros((h, w), dtype=np.int8)
    moisture_map = np.full((h, w), np.nan, dtype=np.float32)
    item_map = np.zeros((h, w), dtype=np.int32)

    # 先用背景光谱铺满整个立方体
    cube = np.broadcast_to(base_spectrum(wl, BACKGROUND), (h, w, b)).copy()

    yy, xx = np.mgrid[0:h, 0:w]  # 像素坐标网格，用来生成椭圆掩码
    classes = np.array([FRESH, AGED, FOREIGN])

    for k in range(1, cfg.n_items + 1):
        cls = int(rng.choice(classes, p=cfg.class_probs))
        #moisture = _sample_moisture(rng, cls)
        moisture = _sample_moisture(rng, cls, cfg)
        # 随机椭圆：中心 (cy, cx)，半轴 (ry, rx)
        cy, cx = rng.uniform(0.15 * h, 0.85 * h), rng.uniform(0.15 * w, 0.85 * w)
        ry, rx = rng.uniform(0.07, 0.13) * h, rng.uniform(0.07, 0.13) * w
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0

        # 物体的理想光谱 + 个体差异（一条平滑的随机曲线，模拟同类物体间的差异）
        #spec = base_spectrum(wl, cls, 0.0 if np.isnan(moisture) else moisture)
        #spec = base_spectrum(wl, cls, 0.0 if np.isnan(moisture) else moisture, cfg.aged_browning)        
        spec = base_spectrum(wl, cls, 0.0 if np.isnan(moisture) else moisture, cfg.aged_browning)     
        if cls == FOREIGN and cfg.subtle_foreign > 0:
            # 混合系数：subtle_foreign=0.7 表示 70% 来自新鲜食品光谱，30% 保留异物特征
            food_spec = base_spectrum(wl, FRESH, 0.8, cfg.aged_browning)
            spec = (cfg.subtle_foreign * food_spec
                    + (1 - cfg.subtle_foreign) * spec).astype(np.float32)        
        individual = rng.normal(0, cfg.item_var_std) * np.cos(
            np.linspace(0, np.pi * rng.uniform(0.5, 2.0), b)
        )
        spec = np.clip(spec + individual, 0.01, 1.0).astype(np.float32)

        cube[mask] = spec
        label_map[mask] = cls
        moisture_map[mask] = moisture
        # 全局唯一物体 ID：scene_id * 1000 + k，跨场景不重复
        item_map[mask] = scene_id * 1000 + k

    # ---- 施加真实感干扰（对每个像素独立采样）----
    # 乘性散射：每个像素整条光谱乘以同一个系数，模拟光照角度/表面粗糙度不同
    scatter = rng.normal(1.0, cfg.scatter_std, size=(h, w, 1)).astype(np.float32)
    # 加性偏移：每个像素整条光谱加上同一个常数
    offset = rng.normal(0.0, cfg.offset_std, size=(h, w, 1)).astype(np.float32)
    # 传感器噪声：每个像素每个波段独立
    #noise = rng.normal(0.0, cfg.noise_std, size=(h, w, b)).astype(np.float32)
    #cube = cube * scatter + offset + noise
    noise = rng.normal(0.0, cfg.noise_std, size=(h, w, b)).astype(np.float32)
    # 线性斜率漂移：每个像素一个随机斜率 × 一条 -0.5~0.5 的斜坡（模拟表面倾斜/光程变化）
    # 放在 noise 之后抽取随机数，保证 slope_std=0 时与旧数据逐位一致
    slope = rng.normal(0.0, cfg.slope_std, size=(h, w, 1)).astype(np.float32)
    ramp = np.linspace(-0.5, 0.5, b, dtype=np.float32)
    cube = cube * scatter + offset + noise + slope * ramp
    return Scene(
        cube=cube.astype(np.float32),
        label_map=label_map,
        moisture_map=moisture_map,
        item_map=item_map,
        wavelengths=wl,
        scene_id=scene_id,
        meta={"n_items": cfg.n_items, "seed": cfg.seed + scene_id},
    )


def generate_dataset(n_scenes: int, cfg: SimConfig | None = None) -> list[Scene]:
    """生成 n_scenes 个不同的场景，返回列表。"""
    cfg = cfg or SimConfig()
    return [generate_scene(cfg, scene_id=i) for i in range(n_scenes)]

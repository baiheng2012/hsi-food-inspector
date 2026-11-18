"""
步骤 7a：profiling。把融合引擎每行的耗时拆成 7 段，找出热点；
同时量化背景阈值的代价（真值是食品，却被判成背景的像素比例）。

运行：uv run python scripts/profile_streaming.py
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dataclasses
import gc
import time
from pathlib import Path

import numpy as np
import yaml

from hsi.data.simulator import generate_dataset
from hsi.evaluation import SIM_CONFIGS
from hsi.streaming.bench import collect_lines
from hsi.streaming.engine import FusedEngine, _empty_result, fit_models

STAGES = ("mask", "alloc", "index", "cls", "reg", "maha", "writeback")
pc = time.perf_counter_ns


def staged(eng: FusedEngine, ln: np.ndarray):
    """与 FusedEngine.process_line 做同样的事，但在每段之间打点。返回 (各段纳秒, 前景像素数)。"""
    t = [pc()]
    x = np.asarray(ln, dtype=eng.dtype)
    fg = x.mean(axis=1) > eng.bg_threshold
    t.append(pc())                                              # mask
    out = _empty_result(x.shape[0])
    t.append(pc())                                              # alloc
    n = int(fg.sum())
    if n == 0:
        return np.concatenate([np.diff(t), np.zeros(5, dtype=np.int64)]), 0
    xf = x[fg]
    t.append(pc())                                              # index
    lab = eng.classes[np.argmax(xf @ eng.A_cls + eng.b_cls, axis=1)]
    t.append(pc())                                              # cls
    mo = xf @ eng.A_reg + eng.b_reg
    t.append(pc())                                              # reg
    d = xf - eng.loc
    an = np.einsum("ij,ij->i", d @ eng.prec, d)
    t.append(pc())                                              # maha
    out.label[fg], out.moisture[fg], out.anomaly[fg] = lab, mo, an
    t.append(pc())                                              # writeback
    return np.diff(t), n


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    eng = FusedEngine.from_sklearn(fit_models(params))
    cfg = dataclasses.replace(SIM_CONFIGS[params["data"]["config"]], seed=1000)
    scenes = generate_dataset(10, cfg)
    lines = collect_lines(scenes)

    for ln in lines[:100]:                                      # 预热
        staged(eng, ln)
    gc.disable()
    rec = [staged(eng, ln) for ln in lines]
    gc.enable()
    T = np.stack([r[0] for r in rec]) / 1e3                     # 微秒
    nfg = np.array([r[1] for r in rec])

    med = np.median(T, axis=0)
    print(f"每行前景像素数：平均 {nfg.mean():.1f}，最大 {nfg.max()}，"
          f"无前景的行占 {np.mean(nfg == 0) * 100:.1f}%（该行只付掩码+分配）")
    print(f"\n{'阶段':10s} {'中位数 µs':>10s} {'占比':>7s}   （占比 = 该段中位数 / 各段中位数之和）")
    for name, m in zip(STAGES, med):
        print(f"{name:10s} {m:10.2f} {m / med.sum() * 100:6.1f}%")
    print(f"{'合计':10s} {med.sum():10.2f}")

    # 背景阈值的代价：真值是食品/异物，但行均值 <= 阈值，被当成背景
    miss = tot = 0
    for s in scenes:
        keep = s.cube.mean(axis=2) > eng.bg_threshold
        truth_fg = s.label_map != 0
        miss += int((truth_fg & ~keep).sum())
        tot += int(truth_fg.sum())
    print(f"\n背景阈值 {eng.bg_threshold} 的代价：真前景像素被误当背景 {miss}/{tot} = "
          f"{miss / tot * 100:.2f}%")


if __name__ == "__main__":
    main()

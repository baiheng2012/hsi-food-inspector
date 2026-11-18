"""
基准工具 (bench.py)
====================
 6/7 共用的计时、汇总、一致性对比、精度评估函数。
注意：线程数环境变量（OMP_NUM_THREADS 等）必须在 import numpy 之前设置，
所以由各脚本自己在顶部设置，这个模块不管。
"""

from __future__ import annotations

import gc
import time

import numpy as np

from hsi.streaming.linescan import iter_lines


def collect_lines(scenes) -> list[np.ndarray]:
    """把所有场景展开成一个行列表，计时时不把生成器开销算进去。"""
    return [ln for s in scenes for ln in iter_lines(s.cube)]


def measure(proc, lines: list[np.ndarray], warmup: int = 50) -> np.ndarray:
    """逐行计时，返回每行耗时（毫秒）。先预热；计时期间关闭 GC，避免垃圾回收造成尖刺。"""
    for ln in lines[:warmup]:
        proc.process_line(ln)
    times = np.empty(len(lines), dtype=np.int64)
    gc.disable()
    try:
        for i, ln in enumerate(lines):
            t0 = time.perf_counter_ns()
            proc.process_line(ln)
            times[i] = time.perf_counter_ns() - t0
    finally:
        gc.enable()
    return times / 1e6


def summarize(ms: np.ndarray, budget_ms: float) -> dict:
    """把每行耗时汇总成 p50/p95/p99/max、吞吐、超预算比例。"""
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "throughput_lines_per_s": float(1000.0 / ms.mean()),
        "over_budget_frac": float((ms > budget_ms).mean()),
    }


def compare(ref, other, scene) -> dict:
    """
    other 与参考实现 ref 在同一场景上的输出差异。
    水分/异常分数只在两边都有有限值的像素上比较（CNN 版不输出这两项，会自然跳过）。
    """
    same = total = 0
    dm = da = 0.0
    for ln in iter_lines(scene.cube):
        a, b = ref.process_line(ln), other.process_line(ln)
        same += int((a.label == b.label).sum())
        total += len(a.label)
        both = np.isfinite(a.moisture) & np.isfinite(b.moisture)
        if both.any():
            dm = max(dm, float(np.abs(a.moisture[both] - b.moisture[both]).max()))
        both = np.isfinite(a.anomaly) & np.isfinite(b.anomaly)
        if both.any():
            rel = np.abs(a.anomaly[both] - b.anomaly[both]) / (np.abs(a.anomaly[both]) + 1e-9)
            da = max(da, float(rel.max()))
    return {"label_agree": same / total, "max_abs_moisture_diff": dm,
            "max_rel_anomaly_diff": da}


def accuracy(engine, scenes) -> dict:
    """
    留出场景上对照真值：
    foreground_acc    : 前景像素分类准确率
    background_ok     : 背景像素被正确当作背景的比例
    fg_missed_as_bg   : 真值是食品/异物，却被背景阈值判成背景的比例（阈值的代价）
    """
    ok_fg = n_fg = ok_bg = n_bg = missed = 0
    for s in scenes:
        for i, ln in enumerate(iter_lines(s.cube)):
            pred, truth = engine.process_line(ln).label, s.label_map[i]
            fg = truth != 0
            ok_fg += int((pred[fg] == truth[fg]).sum())
            missed += int((pred[fg] == 0).sum())
            n_fg += int(fg.sum())
            ok_bg += int((pred[~fg] == 0).sum())
            n_bg += int((~fg).sum())
    return {"foreground_acc": ok_fg / n_fg, "background_ok": ok_bg / n_bg,
            "fg_missed_as_bg": missed / n_fg}

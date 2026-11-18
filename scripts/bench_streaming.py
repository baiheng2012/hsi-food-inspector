"""
步骤 6 基准：逐行计时，对比 sklearn 逐行处理 与 融合 NumPy 引擎。

指标：每行耗时 p50 / p95 / p99 / max（毫秒）、吞吐（行/秒）、超预算比例。
并验证：融合版与 sklearn 输出一致；融合版在留出场景上的分类准确率。

运行：uv run python scripts/bench_streaming.py [--budget-ms 5] [--scenes 10]
输出：docs/results/step6_latency.json, docs/figures/step6_latency.png
"""

import os

# 必须在 import numpy 之前设置：小矩阵乘法用多线程反而增加抖动
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from hsi.data.simulator import generate_dataset
from hsi.evaluation import SIM_CONFIGS
from hsi.streaming.engine import FusedEngine, SklearnProcessor, fit_models
from hsi.streaming.linescan import iter_lines


def measure(proc, scenes, warmup: int = 50) -> np.ndarray:
    """逐行计时，返回每行耗时（毫秒）。先预热，计时期间关闭 GC。"""
    lines = [ln for s in scenes for ln in iter_lines(s.cube)]  # 先取出，避免计入生成器开销
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
    return {
        "lines": len(ms),
        "mean_ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "throughput_lines_per_s": float(1000.0 / ms.mean()),
        "over_budget_frac": float((ms > budget_ms).mean()),
    }


def check_equivalence(fused, sk, scene) -> dict:
    """融合版与 sklearn 在同一场景上的输出差异。"""
    same, total, dm, da = 0, 0, 0.0, 0.0
    for ln in iter_lines(scene.cube):
        a, b = fused.process_line(ln), sk.process_line(ln)
        same += int((a.label == b.label).sum())
        total += len(a.label)
        fg = b.label != 0
        if fg.any():
            dm = max(dm, float(np.abs(a.moisture[fg] - b.moisture[fg]).max()))
            rel = np.abs(a.anomaly[fg] - b.anomaly[fg]) / (np.abs(b.anomaly[fg]) + 1e-9)
            da = max(da, float(rel.max()))
    return {"label_agree": same / total, "max_abs_moisture_diff": dm,
            "max_rel_anomaly_diff": da}


def accuracy(engine, scenes) -> dict:
    """留出场景上，对照真值：前景像素分类准确率、背景像素被正确当作背景的比例。"""
    ok_fg = n_fg = ok_bg = n_bg = 0
    for s in scenes:
        for i, ln in enumerate(iter_lines(s.cube)):
            pred, truth = engine.process_line(ln).label, s.label_map[i]
            fg = truth != 0
            ok_fg += int((pred[fg] == truth[fg]).sum())
            n_fg += int(fg.sum())
            ok_bg += int((pred[~fg] == 0).sum())
            n_bg += int((~fg).sum())
    return {"foreground_acc": ok_fg / n_fg, "background_ok": ok_bg / n_bg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-ms", type=float, default=5.0, help="每行延迟预算")
    ap.add_argument("--scenes", type=int, default=10, help="用于计时的留出场景数")
    args = ap.parse_args()

    params = yaml.safe_load(Path("params.yaml").read_text())
    print("训练部署模型（全量数据）...")
    models = fit_models(params)
    sk = SklearnProcessor(models)
    fused = FusedEngine.from_sklearn(models)

    # 留出场景：换一个随机种子，保证不与训练场景重复
    cfg = dataclasses.replace(SIM_CONFIGS[params["data"]["config"]], seed=1000)
    scenes = generate_dataset(args.scenes, cfg)
    print(f"留出场景 {len(scenes)} 个，每个 {scenes[0].cube.shape[0]} 行，"
          f"每行 {scenes[0].cube.shape[1]} 像素 x {scenes[0].cube.shape[2]} 波段")

    eq = check_equivalence(fused, sk, scenes[0])
    print(f"\n[一致性] 标签一致率 {eq['label_agree']:.5f}，"
          f"水分最大绝对差 {eq['max_abs_moisture_diff']:.2e}，"
          f"异常分数最大相对差 {eq['max_rel_anomaly_diff']:.2e}")
    acc = accuracy(fused, scenes)
    print(f"[准确率] 前景像素 {acc['foreground_acc']:.4f}，背景判对 {acc['background_ok']:.4f}")

    results, all_ms = {}, {}
    for name, proc in (("sklearn", sk), ("fused_numpy", fused)):
        ms = measure(proc, scenes)
        all_ms[name] = ms
        results[name] = summarize(ms, args.budget_ms)
        r = results[name]
        print(f"\n[{name}] p50 {r['p50_ms']:.3f}  p95 {r['p95_ms']:.3f}  p99 {r['p99_ms']:.3f}  "
              f"max {r['max_ms']:.3f} ms | {r['throughput_lines_per_s']:.0f} 行/秒 | "
              f"超预算 {r['over_budget_frac'] * 100:.2f}%")
    speedup = results["sklearn"]["p50_ms"] / results["fused_numpy"]["p50_ms"]
    print(f"\n融合相对 sklearn 的 p50 加速比：{speedup:.1f}x（预算 {args.budget_ms} ms/行）")

    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    Path("docs/results/step6_latency.json").write_text(json.dumps(
        {"budget_ms": args.budget_ms, "equivalence": eq, "accuracy": acc,
         "latency": results, "speedup_p50": speedup}, indent=2))

    # 图：延迟的经验累积分布（越靠左越快；看尾部是否越过预算线）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, ms in all_ms.items():
        xs = np.sort(ms)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=name)
    ax.axvline(args.budget_ms, color="r", ls="--", label=f"budget {args.budget_ms} ms")
    ax.set_xscale("log")
    ax.set_xlabel("per-line latency (ms, log scale)")
    ax.set_ylabel("cumulative fraction of lines")
    ax.set_title("Streaming latency distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/figures/step6_latency.png", dpi=130)
    print("已保存 docs/results/step6_latency.json 和 docs/figures/step6_latency.png")


if __name__ == "__main__":
    main()

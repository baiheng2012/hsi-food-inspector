"""
步骤 7b：优化基准。把各个版本放在同一把尺子下：延迟、与参考的一致性、留出场景精度。

三任务组（分类+水分+异物）：sklearn / fused_f64(步骤6) / opt_f64 / opt_f32 / opt_f32+reuse / onnx_fused
CNN 组（只做分类）        ：cnn_torch / cnn_onnx / cnn_onnx_int8

运行：uv run python scripts/optimize_streaming.py [--repeats 3] [--cnn-epochs 15]
输出：docs/results/step7_benchmark.{json,md}  docs/figures/step7_benchmark.png
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from hsi.data.simulator import generate_dataset
from hsi.evaluation import SIM_CONFIGS, build_data
from hsi.models.baselines import make_baseline
from hsi.streaming.bench import accuracy, collect_lines, compare, measure, summarize
from hsi.streaming.engine import FusedEngine, SklearnProcessor, fit_models
from hsi.streaming.engine_opt import OptimizedEngine
from hsi.streaming.onnx_tools import (
    FusedTorch,
    LabelOnlyProcessor,
    OnnxFusedEngine,
    build_cnn_module,
    export_onnx,
    make_session,
    ort_logits_fn,
    quantize_linear_int8,
    torch_logits_fn,
)

torch.set_num_threads(1)  # 与 NumPy 基线公平：单线程


def median_summary(summaries: list[dict]) -> dict:
    """多次重复测量，对每个统计量取中位数，削弱系统抖动。"""
    return {k: float(np.median([s[k] for s in summaries])) for k in summaries[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-ms", type=float, default=5.0)
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cnn-epochs", type=int, default=15)
    args = ap.parse_args()
    params = yaml.safe_load(Path("params.yaml").read_text())
    mdir = Path("models")

    # ---------- 三任务组 ----------
    print("训练三任务模型 ...")
    models = fit_models(params)
    base = FusedEngine.from_sklearn(models)
    n_bands = base.A_cls.shape[0]
    opt32 = OptimizedEngine(base, np.float32)
    export_onnx(FusedTorch(opt32), torch.zeros(16, n_bands), mdir / "fused.onnx", ["x"], ["z", "maha"])

    procs = {   # 名称 -> (处理器, 任务组, 参考组)
        "sklearn": SklearnProcessor(models),
        "fused_f64 (step6)": base,
        "opt_f64 (merged)": OptimizedEngine(base, np.float64),
        "opt_f32": opt32,
        "opt_f32+reuse": OptimizedEngine(base, np.float32, reuse_buffers=True),
        "onnx_fused_f32": OnnxFusedEngine(mdir / "fused.onnx", opt32),
    }
    group = {n: "3-task" for n in procs}
    ref_of = {n: "fused_f64 (step6)" for n in procs}

    # ---------- CNN 组 ----------
    print(f"训练 1D-CNN（{args.cnn_epochs} 轮，全量前景像素）...")
    X, y, _, _ = build_data(params["data"])
    pipe = make_baseline("cnn", "raw", epochs=args.cnn_epochs).fit(X, y)
    module, classes = build_cnn_module(pipe)
    export_onnx(module, torch.zeros(16, n_bands), mdir / "cnn.onnx", ["x"], ["logits"])
    cnn_procs = {
        "cnn_torch": LabelOnlyProcessor(torch_logits_fn(module), classes),
        "cnn_onnx_fp32": LabelOnlyProcessor(
            ort_logits_fn(make_session(mdir / "cnn.onnx")), classes),
    }
    try:
        quantize_linear_int8(mdir / "cnn.onnx", mdir / "cnn_int8.onnx")
        cnn_procs["cnn_onnx_int8"] = LabelOnlyProcessor(
            ort_logits_fn(make_session(mdir / "cnn_int8.onnx")), classes)
    except Exception as e:  # noqa: BLE001  量化/加载失败的类型因 onnxruntime 版本而异，如实打印
        print(f"[警告] int8 量化版不可用，已跳过：{type(e).__name__}: {e}")
    for n, p in cnn_procs.items():
        procs[n], group[n], ref_of[n] = p, "cnn (cls only)", "cnn_torch"

    # ---------- 留出场景 ----------
    cfg = dataclasses.replace(SIM_CONFIGS[params["data"]["config"]], seed=1000)
    scenes = generate_dataset(args.scenes, cfg)
    lines = collect_lines(scenes)
    print(f"留出场景 {len(scenes)} 个，共 {len(lines)} 行；预算 {args.budget_ms} ms/行\n")

    # 一致性与精度（只算一次）
    quality = {}
    for n, p in procs.items():
        eq = compare(procs[ref_of[n]], p, scenes[0])
        quality[n] = {**eq, **accuracy(p, scenes)}

    # 延迟：重复在外层、处理器在内层，使系统漂移对各版本影响一致
    runs = {n: [] for n in procs}
    for r in range(args.repeats):
        print(f"计时 {r + 1}/{args.repeats} ...")
        for n, p in procs.items():
            runs[n].append(summarize(measure(p, lines), args.budget_ms))
    lat = {n: median_summary(v) for n, v in runs.items()}

    # ---------- 输出 ----------
    ref_p50 = lat["fused_f64 (step6)"]["p50_ms"]
    rows, md = [], [
        "| 版本 | 任务 | p50 ms | p95 ms | p99 ms | max ms | 相对 step6 p50 | 标签一致率 | "
        "水分最大差 | 异常分数最大相对差 | 前景准确率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in procs:
        L, Q = lat[n], quality[n]
        sp = ref_p50 / L["p50_ms"]
        rows.append({"name": n, "group": group[n], **L, "speedup_vs_step6": sp, **Q})
        md.append(f"| {n} | {group[n]} | {L['p50_ms']:.3f} | {L['p95_ms']:.3f} | {L['p99_ms']:.3f} | "
                  f"{L['max_ms']:.3f} | {sp:.2f}x | {Q['label_agree']:.5f} | "
                  f"{Q['max_abs_moisture_diff']:.1e} | {Q['max_rel_anomaly_diff']:.1e} | "
                  f"{Q['foreground_acc']:.4f} |")
    table = "\n".join(md)
    print("\n" + table)

    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    Path("docs/results/step7_benchmark.md").write_text(table + "\n")
    Path("docs/results/step7_benchmark.json").write_text(
        json.dumps({"budget_ms": args.budget_ms, "repeats": args.repeats, "rows": rows}, indent=2))

    names = [r["name"] for r in rows]
    pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(pos - 0.2, [r["p50_ms"] for r in rows], 0.4, label="p50")
    ax.barh(pos + 0.2, [r["p99_ms"] for r in rows], 0.4, label="p99")
    ax.axvline(args.budget_ms, color="r", ls="--", label=f"budget {args.budget_ms} ms")
    ax.set_yticks(pos, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("per-line latency (ms, log scale)")
    ax.set_title("Streaming latency by implementation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/figures/step7_benchmark.png", dpi=130)
    print("\n已保存 docs/results/step7_benchmark.{json,md} 和 docs/figures/step7_benchmark.png")


if __name__ == "__main__":
    main()

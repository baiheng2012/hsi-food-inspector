"""
演示客户端：把一个留出场景逐行 POST 给服务（模拟相机吐一行、送一行）。
报告：准确率、服务端引擎耗时、客户端往返耗时（含 HTTP + JSON 编解码）。

运行：先启动服务，再 uv run python scripts/demo_client.py [--url http://localhost:8000]
"""

import argparse
import dataclasses
import json
import time
import urllib.request

import numpy as np

from hsi.data.simulator import generate_scene
from hsi.evaluation import SIM_CONFIGS


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--config", default="subtle", choices=list(SIM_CONFIGS))
    args = ap.parse_args()

    scene = generate_scene(dataclasses.replace(SIM_CONFIGS[args.config], seed=1000), 0)
    rtt, engine_ms, ok_fg, n_fg = [], [], 0, 0
    for i in range(scene.cube.shape[0]):
        t0 = time.perf_counter()
        body = post_json(f"{args.url}/predict_line", {"line": scene.cube[i].tolist()})
        rtt.append((time.perf_counter() - t0) * 1e3)
        engine_ms.append(body["latency_ms"])
        pred, truth = np.array(body["label"]), scene.label_map[i]
        fg = truth != 0
        ok_fg += int((pred[fg] == truth[fg]).sum())
        n_fg += int(fg.sum())

    q = lambda a, p: float(np.percentile(a, p))
    print(f"行数 {len(rtt)}，前景像素准确率 {ok_fg / n_fg:.4f}")
    print(f"引擎耗时(服务端) p50 {q(engine_ms, 50):.3f} ms  p99 {q(engine_ms, 99):.3f} ms")
    print(f"往返耗时(客户端) p50 {q(rtt, 50):.3f} ms  p99 {q(rtt, 99):.3f} ms")


if __name__ == "__main__":
    main()

"""
推理服务 (api.py)
==================

端点
----
GET  /health        存活检查（Docker / 负载均衡用）
GET  /info          模型信息：波段数、类别、背景阈值
POST /predict_line  输入一行光谱 (W, B)，返回每个像素的类别、水分、异常分数

设计要点
--------
- create_app(engine=None)：应用工厂。测试时直接注入引擎；生产时在启动阶段
  从 HSI_MODEL_PATH（默认 models/engine.npz）加载，文件不存在就启动失败（快速失败）。
- 处理函数是普通 def：FastAPI 会放进线程池执行。引擎不修改自身状态
  （reuse_buffers=False），多线程并发是安全的。
- JSON 不能表示 NaN，背景像素的水分/异常分数用 null 表示。
- 返回的 latency_ms 只含引擎计算，不含 HTTP 和 JSON 解析的开销。
"""

import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from hsi.data.simulator import CLASS_NAMES
from hsi.streaming.engine_opt import OptimizedEngine

MAX_PIXELS = 4096  # 一行最多多少像素，防止超大请求耗尽内存


class LineRequest(BaseModel):
    line: list[list[float]] = Field(
        ..., min_length=1, max_length=MAX_PIXELS,
        description="一行光谱，形状 (W, B)：W 个像素，每个像素 B 个波段的反射率",
    )


class LineResponse(BaseModel):
    label: list[int]                 # 0 背景 / 1 新鲜 / 2 陈旧 / 3 异物
    label_name: list[str]
    moisture: list[float | None]     # 背景为 null
    anomaly: list[float | None]      # 马氏距离平方，越大越像异物；背景为 null
    latency_ms: float                # 仅引擎计算耗时


def _json_floats(a: np.ndarray) -> list[float | None]:
    """NaN/Inf -> None（标准 JSON 不允许 NaN）。"""
    return [float(v) if math.isfinite(v) else None for v in a.tolist()]


def create_app(engine: OptimizedEngine | None = None, model_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            path = Path(model_path or os.environ.get("HSI_MODEL_PATH", "models/engine.npz"))
            if not path.exists():
                raise RuntimeError(f"找不到模型文件 {path}，请先运行 scripts/export_model.py")
            app.state.engine = OptimizedEngine.load(path)
        yield

    app = FastAPI(title="HSI Food Inspector", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "n_bands": request.app.state.engine.n_bands}

    @app.get("/info")
    def info(request: Request):
        eng = request.app.state.engine
        return {
            "n_bands": eng.n_bands,
            "classes": {int(c): CLASS_NAMES[int(c)] for c in eng.classes},
            "dtype": eng.dtype.name,
            "background_threshold": eng.bg_threshold,
            "max_pixels_per_line": MAX_PIXELS,
        }

    @app.post("/predict_line", response_model=LineResponse)
    def predict_line(req: LineRequest, request: Request):
        eng = request.app.state.engine
        try:
            x = np.asarray(req.line, dtype=np.float32)
        except ValueError as e:  # 各行长度不一致（ragged）
            raise HTTPException(422, f"line 的各行长度必须一致：{e}") from e
        if x.ndim != 2 or x.shape[1] != eng.n_bands:
            raise HTTPException(422, f"期望形状 (W, {eng.n_bands})，实际 {list(x.shape)}")
        if not np.isfinite(x).all():
            raise HTTPException(422, "输入含 NaN 或 Inf")

        t0 = time.perf_counter()
        r = eng.process_line(x)
        ms = (time.perf_counter() - t0) * 1e3

        labels = r.label.tolist()
        return LineResponse(
            label=labels,
            label_name=[CLASS_NAMES[i] for i in labels],
            moisture=_json_floats(r.moisture),
            anomaly=_json_floats(r.anomaly),
            latency_ms=ms,
        )

    return app


# uvicorn hsi.serving.api:app 使用的实例。模型在启动阶段才加载，导入本模块不需要模型文件。
app = create_app()

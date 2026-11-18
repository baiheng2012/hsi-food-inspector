"""
API 测试：守护"接口输出与引擎直接调用一致""非法输入返回 422""模型文件往返无损"。
"""

import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hsi.data.simulator import HARD_CONFIG, generate_scene
from hsi.serving.api import create_app
from hsi.streaming.engine import FusedEngine, fit_models
from hsi.streaming.engine_opt import OptimizedEngine

PARAMS = {
    "data": {"config": "hard", "n_scenes": 6, "per_item": 30, "seed": 0},
    "models": {
        "classifier": {"name": "plsda", "prep": "raw"},
        "regressor": {"name": "ridge", "prep": "raw"},
        "detector": {"name": "mahalanobis", "prep": "raw"},
    },
}


@pytest.fixture(scope="module")
def engine():
    return OptimizedEngine(FusedEngine.from_sklearn(fit_models(PARAMS)), np.float32)


@pytest.fixture(scope="module")
def scene():
    return generate_scene(dataclasses.replace(HARD_CONFIG, seed=500), 0)


@pytest.fixture()
def client(engine):
    return TestClient(create_app(engine=engine))  # 注入引擎，不需要模型文件


def test_health_and_info(client, engine):
    assert client.get("/health").json() == {"status": "ok", "n_bands": engine.n_bands}
    info = client.get("/info").json()
    assert info["n_bands"] == engine.n_bands
    assert set(info["classes"].values()) <= {"fresh", "aged", "foreign"}


def test_predict_line_matches_engine(client, engine, scene):
    row = int(np.argmax((scene.label_map != 0).sum(axis=1)))  # 前景像素最多的一行
    line = scene.cube[row]
    resp = client.post("/predict_line", json={"line": line.tolist()})
    assert resp.status_code == 200
    body, direct = resp.json(), engine.process_line(line)
    assert body["label"] == direct.label.tolist()
    fg = direct.label != 0
    assert fg.any()
    got = np.array([body["moisture"][i] for i in np.flatnonzero(fg)])
    assert np.allclose(got, direct.moisture[fg], atol=1e-5)
    assert body["latency_ms"] >= 0


def test_background_line_returns_nulls(client, engine):
    line = np.full((8, engine.n_bands), 0.05).tolist()
    body = client.post("/predict_line", json={"line": line}).json()
    assert body["label"] == [0] * 8
    assert body["moisture"] == [None] * 8 and body["anomaly"] == [None] * 8


def test_wrong_band_count_is_422(client):
    assert client.post("/predict_line", json={"line": [[0.1] * 7]}).status_code == 422


def test_ragged_line_is_422(client, engine):
    line = [[0.1] * engine.n_bands, [0.1] * 3]
    assert client.post("/predict_line", json={"line": line}).status_code == 422


def test_empty_line_is_422(client):
    assert client.post("/predict_line", json={"line": []}).status_code == 422


def test_save_load_roundtrip(engine, scene, tmp_path):
    path = tmp_path / "e.npz"
    engine.save(path)
    loaded = OptimizedEngine.load(path)
    assert loaded.dtype == engine.dtype and loaded.n_bands == engine.n_bands
    for row in (10, 40, 90):
        a, b = engine.process_line(scene.cube[row]), loaded.process_line(scene.cube[row])
        assert np.array_equal(a.label, b.label)
        assert np.allclose(a.moisture, b.moisture, equal_nan=True)


def test_lifespan_loads_model_from_env(engine, tmp_path, monkeypatch):
    path = tmp_path / "e.npz"
    engine.save(path)
    monkeypatch.setenv("HSI_MODEL_PATH", str(path))
    with TestClient(create_app()) as c:   # with 语句才会触发启动事件
        assert c.get("/health").status_code == 200


def test_missing_model_file_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("HSI_MODEL_PATH", str(tmp_path / "nope.npz"))
    with pytest.raises(RuntimeError), TestClient(create_app()):
        pass

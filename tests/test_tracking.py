import mlflow

from hsi.tracking import flatten, log_run


def test_flatten_nested():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_log_run_roundtrip(tmp_path, monkeypatch):
    # 切到临时目录：MLflow 默认把产物写到当前目录下的 mlruns，不要污染项目
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/t.db"
    run_id = log_run({"a": {"b": 1}}, {"m": 0.5}, tracking_uri=uri, experiment="t")
    run = mlflow.tracking.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.data.params["a.b"] == "1"
    assert run.data.metrics["m"] == 0.5

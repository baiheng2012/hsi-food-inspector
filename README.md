# HSI Food Inspector

[![CI](https://github.com/<YOUR_USERNAME>/hsi-food-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_USERNAME>/hsi-food-inspector/actions)

An end-to-end ML system for **hyperspectral food inspection** on a simulated conveyor belt:
quality classification, moisture regression and foreign-object detection, served by a
real-time streaming engine, tracked with DVC + MLflow, and guarded by a model-regression gate in CI.

> **Data note.** Every result below comes from a physics-inspired **simulator** (NIR 900–1700 nm,
> 100 bands, multiplicative scatter, baseline slope, sensor noise), not from real hyperspectral
> data. The numbers show how the methods behave under controlled difficulty, not how they would
> perform in production. The latency budget (5 ms/line, i.e. 200 lines/s) is an assumption, not
> the spec of a real camera.

## What's in here

- **Leak-free validation.** All cross-validation is grouped by physical object
  (`StratifiedGroupKFold`), with an assertion that no object appears in both train and test.
- **Three tasks** on one data layer: 4-class pixel classification, moisture regression,
  foreign-object detection trained on *normal* food only.
- **Streaming inference** that mimics a pushbroom camera (one `(W, B)` line at a time), with
  p50/p95/p99 latency measurements.
- **MLOps:** DVC pipeline, MLflow experiment tracking, and a CI gate that fails when metrics regress.
- **Serving:** FastAPI + Docker, built reproducibly from `uv.lock`.

## Results

### 1. Random CV leaks; grouped CV does not (moisture regression, R²)

| Model + preprocessing | Random CV | Grouped CV | Gap |
|---|---|---|---|
| kNN + SNV | 0.903 | 0.800 | 0.103 |
| kNN + MSC | 0.898 | 0.798 | 0.100 |
| Ridge, raw | 0.961 | 0.959 | 0.002 |

Higher-capacity models "recognise" objects they have seen pixels of, so they leak more.
Grouped CV also exposed a much larger fold-to-fold variance (kNN+SNV std 0.002 → 0.089).

### 2. Classification (hard simulation, 5-fold grouped CV, pixel macro-F1)

| Model | Best preprocessing | F1 |
|---|---|---|
| PLS-DA | raw | 0.969 |
| Logistic regression | raw / SNV | 0.967 |
| SVM (RBF) | SNV | 0.962 |
| Random forest | SG-1st-derivative + SNV | 0.943 (raw: 0.840) |
| 1D-CNN | raw | TODO: re-run `scripts/exp_classification.py` |

- PLS-DA, logistic regression, SVM and the CNN are within noise of each other (fold std ≈ 0.005);
  the CNN costs roughly 200x more training time than PLS-DA.
- Random forest is sensitive to multiplicative scatter: preprocessing lifts it from 0.84 to 0.94.
- 2nd-derivative + SNV (window 11) hurts every model (0.75–0.82), most likely noise amplification (not verified).
- Remaining errors are almost entirely fresh <-> aged; foreign objects are recall ≈ 1.0.

### 3. Preprocessing must be chosen per task

| Moisture regression, fresh class only (Ridge) | R² | RMSE |
|---|---|---|
| raw | 0.710 | 0.029 |
| SNV | -0.024 | 0.055 |

SNV helps classification (shape) but removes the amplitude that carries the moisture signal.
The model-regression test in CI catches exactly this mistake (see `tests/test_model_regression.py`).

### 4. Foreign-object detection (trained on normal food only, "subtle" foreign objects)

| Detector + preprocessing | AUC | Detection rate @ 1% false-alarm |
|---|---|---|
| Mahalanobis, raw | 0.986 | **0.859** |
| SAM, SG-1st-derivative + SNV | 0.985 | 0.478 |
| Isolation Forest, SG-1st-derivative + SNV | 0.920 | 0.096 |

AUC alone is misleading: Mahalanobis + SG1 + SNV has AUC 0.947 but only 5.8% detection at 1% false alarms.

### 5. Real-time inference (1280 lines, 128 px × 100 bands, single thread)

Three tasks per line (classify + moisture + anomaly score) unless noted.

| Implementation | p50 ms | p99 ms |
|---|---|---|
| sklearn, called per line | 1.123 | 2.176 |
| Fused NumPy, float64 | 0.071 | 0.227 |
| **Fused NumPy, float32** | **0.051** | **0.150** |
| ONNX Runtime (same math) | 0.080 | 0.212 |
| 1D-CNN, PyTorch (classification only) | 0.646 | 1.750 |
| 1D-CNN, ONNX fp32 (classification only) | 0.336 | 1.264 |
| 1D-CNN, ONNX int8 (classification only) | 0.347 | 1.331 |

- Fusing PLS-DA, Ridge and the Mahalanobis score into plain matrix operations is mathematically
  exact (label agreement 100%, numeric difference ≤ 1e-5) and about 22x faster than per-line sklearn.
- float32 was the one optimisation that mattered (1.4x). Merging matmuls (1.03x), ONNX Runtime
  (0.9x) and int8 quantisation (no gain) did not help at this matrix size; they are reported anyway.
- Profiling: the Mahalanobis quadratic form (35%) and the background mask (28%) dominate.
- The background threshold misclassifies 0.70% of true foreground pixels as background.

Full table: `docs/results/step7_benchmark.md`.

## Quickstart

```bash
uv sync                                   # exact environment from uv.lock
uv run pytest -q -m "not model"           # unit tests
uv run dvc repro                          # generate data + evaluate (params.yaml)
uv run pytest -m model                    # metrics must not regress vs baselines/
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Reproduce the experiments: `scripts/exp_*.py`, `scripts/bench_streaming.py`, `scripts/optimize_streaming.py`.
The simulator is deterministic, so data regenerates identically from the seeds in `params.yaml`.

### Serve it

```bash
docker build -t hsi-food-inspector .
docker run --rm -p 8000:8000 hsi-food-inspector
# interactive docs: http://localhost:8000/docs
uv run python scripts/demo_client.py      # streams a held-out scene line by line
```

`POST /predict_line` takes one `(W, B)` line and returns per-pixel class, moisture and a
Mahalanobis anomaly score (`null` for background pixels).

## Layout

```
src/hsi/
  data/simulator.py        scene simulator (classes, moisture, scatter, slope, noise)
  preprocessing.py         SNV / MSC / Savitzky-Golay as sklearn transformers
  validation.py            grouped CV, leakage assertion, anomaly CV
  models/                  baselines, 1D-CNN, regressors, anomaly detectors
  evaluation.py            single evaluation entry point (training, tracking and CI share it)
  streaming/               line-scan simulator, fused engines, ONNX tools
  serving/api.py           FastAPI service
scripts/                   experiments, benchmarks, DVC stages
tests/                     unit, model-regression and API tests
```

## Limitations

- Simulated data only; no real hyperspectral dataset has been evaluated yet.
- The fused engine supports `raw` preprocessing only (the best choice for moisture here).
- The API returns a raw anomaly score; a calibrated decision threshold is not implemented.
- HTTP + JSON overhead dominates the API round-trip; the real-time path is the in-process engine.
- The Docker image includes training dependencies (torch, mlflow, dvc) and is larger than necessary.
- The DVC remote in this repo is a local directory, for demonstration only.

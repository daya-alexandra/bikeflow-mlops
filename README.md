# BikeFlow

BikeFlow is a team learning project that will grow into an end-to-end MLOps system for forecasting
hourly urban bicycle-rental demand. It follows the requirements of the
[MLOps course](https://github.com/Discipliny/mlops-course) and uses the
[Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

## Current stage

The repository provides a tested FastAPI boundary, a reviewed model/API contract, CI, a non-root
Docker image, and a reproducible training pipeline that produces a real model artifact.

The API still serves `StubPredictor` by default: swapping it for the trained model is a one-line
change in `bikeflow/api/dependencies.py`, held back until we agree how the artifact reaches the
container. There is no persistence, DVC remote, MLflow, monitoring, orchestration, deployment or UI
yet.

Participant A (`EgorMa1tsev`) owns data, DVC, preprocessing, training, and the ML artifact.
Participant B (`daya-alexandra`) owns the repository, API, tests, CI, Docker, and later platform
infrastructure. The integration agreement is documented in
[`docs/contracts/model_api.md`](docs/contracts/model_api.md).

## Install and run

Python 3.11 is required.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn bikeflow.api.main:app --reload --host 127.0.0.1 --port 8000
```

OpenAPI UI is available at <http://127.0.0.1:8000/docs> and the schema at
<http://127.0.0.1:8000/openapi.json>.

## Verify

```bash
make lint
make test
make docker-build
```

The equivalent cross-platform commands are `ruff check .`, `ruff format --check .`, `pytest`, and
`docker build --tag bikeflow:local .`.

## Example prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_time": "2026-07-15T08:00:00+09:00",
    "temperature_c": 24.5,
    "humidity_pct": 61,
    "wind_speed_m_s": 1.8,
    "visibility_10m": 1800,
    "dew_point_c": 16.4,
    "solar_radiation_mj_m2": 1.2,
    "rainfall_mm": 0,
    "snowfall_cm": 0,
    "holiday": false,
    "functioning_day": true
  }'
```

The response contains the requested time, `predicted_rentals`, and `model_version: "stub-v0"`.

## Training pipeline

The data and model side needs the `ml` extra (pandas, scikit-learn, torch). It is deliberately
separate from the runtime dependencies so the API image does not carry torch.

```bash
python -m pip install -e ".[dev,ml]"
make data      # download from UCI, preprocess, chronological split
make train     # baseline + reference + two networks, then reports/
make evaluate  # metrics for the serving model, with slices
```

Training runs on CPU in about a minute and is deterministic: a clean rerun reproduces every metric
bit for bit.

| Model | val WAPE | test MAE | test WAPE |
| --- | --- | --- | --- |
| seasonal_median (baseline) | 0.566 | 422.9 | 0.497 |
| hgb (reference) | 0.163 | 279.4 | 0.329 |
| **mlp_embedding (serving)** | 0.174 | **180.3** | **0.212** |

The serving model is a PyTorch MLP with entity embeddings; it improves test MAE by 57 % over the
seasonal baseline (R² = 0.805). Architecture, metrics, slices and limitations are in
[`docs/model/model_card.md`](docs/model/model_card.md); the dataset and split are described in
[`docs/model/data_card.md`](docs/model/data_card.md).

Predictions are also available directly, for one observation or many at once:

```python
from bikeflow.ml.inference import Predictor

predictor = Predictor.load("models/model.joblib")
predictor.predict({"date": "2018-12-01", "hour": 18, "season": "Winter", ...})
predictor.predict(list_of_rows)          # batch
predictor.predict("hours.csv")           # file
```

## Docker

```bash
docker build --tag bikeflow:local .
docker run --rm -p 8000:8000 bikeflow:local
```

## Collaboration workflow

`main` is protected after the bootstrap commit. Create a focused branch such as
`feat/model-baseline`, `feat/dvc-pipeline`, `fix/predict-validation`, or `docs/model-contract`; use
[Conventional Commits](https://www.conventionalcommits.org/) and open a pull request. A merge needs
passing CI and one approval. Prefer squash merge; GitHub deletes merged branches automatically.

Do not commit datasets, trained models, binary artifacts, `.env`, credentials, or tokens.

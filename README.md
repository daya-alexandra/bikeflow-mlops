# BikeFlow

BikeFlow is a team learning project that will grow into an end-to-end MLOps system for forecasting
hourly urban bicycle-rental demand. It follows the requirements of the
[MLOps course](https://github.com/Discipliny/mlops-course) and uses the
[Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

## Current stage

This first stage provides a small, tested FastAPI boundary, a preliminary model/API contract, CI,
and a non-root Docker image. The API currently uses a fixed development-only stub model; there is
no real ML model, persistence, DVC remote, MLflow, monitoring, orchestration, deployment, or UI.

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

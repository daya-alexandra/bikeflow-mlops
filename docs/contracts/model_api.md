# Model/API contract

Status: **preliminary**. Participant A must review and approve this contract before a real model is
connected. The current service uses `StubPredictor` (`stub-v0`) only.

## Prediction request

`POST /predict` accepts one JSON object:

| Field | Type | Constraint |
| --- | --- | --- |
| `prediction_time` | ISO 8601 datetime | Time of the requested prediction |
| `temperature_c` | number | -100 to 80 |
| `humidity_pct` | number | 0 to 100 |
| `wind_speed_m_s` | number | Non-negative |
| `visibility_10m` | integer | Non-negative, dataset units of 10 m |
| `dew_point_c` | number | -100 to 80 |
| `solar_radiation_mj_m2` | number | Non-negative |
| `rainfall_mm` | number | Non-negative |
| `snowfall_cm` | number | Non-negative |
| `holiday` | boolean | Whether the date is a holiday |
| `functioning_day` | boolean | Whether rentals operate that day |

`hour` and meteorological `season` are derived centrally from `prediction_time`; clients must not
send them. Unknown request fields are rejected. Participant A must confirm names, units, category
encoding, timezone handling, and all value constraints against the training pipeline.

## Prediction response

```json
{
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "predicted_rentals": 42.0,
  "model_version": "stub-v0"
}
```

`predicted_rentals` is numeric and non-negative. `model_version` identifies the exact inference
artifact.

## Predictor interface

The API depends on the `Predictor` protocol. It accepts exactly one feature mapping, returns one
numeric demand prediction, and exposes an immutable model version. Training, drift detection, and
retraining are intentionally outside this interface. Dependency injection allows the stub to be
replaced later by an sklearn/MLflow adapter without changing endpoint code.

## Artifact and validation requirements

Preprocessing must be saved together with the estimator in the same model artifact so online and
offline transformations cannot diverge. Data must use a chronological train/validation/test split,
without random shuffling, to avoid future-to-past leakage.

Participant A must deliver:

- one reproducible training command;
- the packaged preprocessing plus model artifact;
- an immutable model version;
- validation and test metrics, including the chosen primary metric;
- one control request and its expected prediction.

Drift detection and retraining will be separate components in later stages. They must not be hidden
inside the inference adapter.

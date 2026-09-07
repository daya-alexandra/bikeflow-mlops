# Model/API contract

Status: **reviewed by participant A**, see [`../model/contract_review.md`](../model/contract_review.md).
The contract holds. Names, units, category encoding and the split requirement are confirmed against
the trained pipeline; `day_of_week` was added to the derived features, and value ranges plus
timezone handling still need a joint decision. The service still ships `StubPredictor` (`stub-v0`)
by default; the trained artifact is served through `bikeflow.model.adapter.BikeflowPredictor`.

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

`hour`, `day_of_week` and meteorological `season` are derived centrally from `prediction_time`;
clients must not send them. Unknown request fields are rejected. The derived `season` was verified
against the dataset's own labels and matches on all 8760 rows.

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

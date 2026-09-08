# Contract review by participant A

Reviewed against the trained pipeline on the branch `feat/training-pipeline-and-model`.
Reference: [`docs/contracts/model_api.md`](../contracts/model_api.md).

Verdict: **the contract holds**, with one required change and three items that need a decision
before we call the schema final.

## Confirmed

| Item | Status |
| --- | --- |
| Chronological split, no shuffling | Confirmed. train `2017-12-01…2018-07-31`, validation `2018-08-01…2018-09-30`, test `2018-10-01…2018-11-30` |
| Preprocessing packaged with the estimator | Confirmed. One joblib bundle holds estimator, feature order, category levels and the scaler |
| Immutable model version | Added. `metadata.model_version`, e.g. `mlp_embedding-373339b7-20260907T181012Z` |
| One reproducible training command | `make train`. A clean rerun reproduces every metric bit for bit |
| `predicted_rentals` numeric and non-negative | Confirmed. Softplus output plus an explicit clip |
| Season derived from `prediction_time` | **Verified against the data**: your `season_from_month` matches the dataset's own `Seasons` labels on all 8760 rows. Safe to derive |
| Units | Confirmed. `visibility_10m` really is in units of 10 m, `solar_radiation` in MJ/m², `snowfall` in cm |
| Field names | Mapped in `bikeflow/model/adapter.py::FIELD_ALIASES`. No renaming needed on your side |
| Category encoding | The model owns it. `season` arrives lowercase from the API and is normalised inside the feature layer |
| Training/drift outside the interface | Confirmed. The adapter only loads and predicts |

## Required change (already in this branch)

**`day_of_week` must be derived alongside `hour` and `season`.**

The model uses the weekday, and it matters: on test, MAE is 143 on weekends against 194 on
weekdays. Same request, `2018-12-03T18:00` (Monday) predicts 621 rentals while `2018-12-01T18:00`
(Saturday) predicts 344.

`to_features()` previously dropped `prediction_time` before the weekday could be derived, so the
information was unrecoverable downstream. Added one line, exactly parallel to the existing `hour`
and `season` derivation. **The client-facing request schema does not change** — callers still send
only `prediction_time`.

## Needs your decision

**1. Value ranges are wider in the API than in the model.**

| Field | API accepts | Model accepts | Training data |
| --- | --- | --- | --- |
| `temperature_c` | −100 … 80 | −40 … 50 | −17.8 … 39.4 |
| `dew_point_c` | −100 … 80 | −40 … 40 | −30.6 … 27.2 |
| `visibility_10m` | ≥ 0, no maximum | 0 … 2000 | 27 … 2000 |
| `humidity_pct` | 0 … 100 | 0 … 100 | 0 … 98 |

A request that passes your validation can therefore be rejected inside the model layer, which
currently surfaces as a 500. Two options: relax the model bounds to match yours, or tighten yours
to the model's. I lean towards tightening the API — a −80 °C reading in Seoul is a broken sensor,
not a prediction request — but this is a product decision and the error handling is yours.
`visibility_10m` has a real physical maximum of 2000 in this dataset.

**2. Timezone handling.** The dataset is local Seoul time with no offset, and the model uses only
the hour and weekday. Your example sends `+09:00`. Right now a naive datetime and an aware one in
another offset would produce different hours. Suggestion: require an offset and normalise to
`Asia/Seoul` before deriving hour, weekday and season.

**3. Out-of-distribution seasons.** The model has never seen `Autumn` during training (see the
data card): the dataset is exactly one year and autumn falls entirely in the test window. The
network handles this deterministically — unseen category levels are zeroed rather than left at
random initialisation — but predictions for autumn lean on weather, not on the season label. Worth
a note in the model card rather than a code change.

## Control request and expected prediction

For the smoke test asked for in the contract, using
`model_version = mlp_embedding-373339b7-20260907T181012Z`:

```json
{
  "prediction_time": "2018-12-01T18:00:00",
  "temperature_c": 3.5, "humidity_pct": 45, "wind_speed_m_s": 1.2,
  "visibility_10m": 2000, "dew_point_c": -7.0, "solar_radiation_mj_m2": 0.0,
  "rainfall_mm": 0.0, "snowfall_cm": 0.0, "holiday": false, "functioning_day": true
}
```

Expected `predicted_rentals`: **344.1** (±0.1).

With `"functioning_day": false` and everything else identical, the expected value is exactly
**0.0** — closed hours bypass the model entirely, because in the data a non-operating hour always
has zero rentals.

## Switching the API off the stub

`get_predictor` in `bikeflow/api/dependencies.py` is the only place that needs to change:

```python
from bikeflow.model.adapter import BikeflowPredictor


@lru_cache
def get_predictor() -> Predictor:
    return BikeflowPredictor()  # path from params.yaml: models/model.joblib
```

I have left it on `StubPredictor` in this branch so the API image is not forced to carry torch
before you have decided how the artifact reaches the container. The adapter imports its ML
dependencies lazily, so nothing breaks either way.

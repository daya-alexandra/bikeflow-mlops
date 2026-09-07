# Model card — BikeFlow demand forecaster

## What it does

Predicts the number of bicycle rentals in Seoul for one hour, given the calendar position of that
hour and the weather expected during it. One request describes one hour and is self-contained: no
history of past rentals is needed, which is what lets the API stay stateless.

Regression on counts. Serving artifact: `models/model.joblib`.

## Architecture

A PyTorch MLP with entity embeddings for the categorical inputs.

```
hour   -> Embedding(24, 4)
weekday-> Embedding(7,  3)   ->  concat 9
season -> Embedding(4,  2)
9 numeric features, standardised (mean/std fitted on train only)
                    -> concat 18
   Linear(18, 128) + ReLU + Dropout(0.1)
   Linear(128, 64) + ReLU
   Linear(64, 1)   + Softplus          (guarantees a non-negative count)
```

10 878 parameters. Loss `PoissonNLL`, Adam (lr 3e-3, weight decay 1e-4), batch 256, early stopping
on validation MAE with patience 25. CPU only; training takes under a minute.

Embeddings are used rather than one-hot so the network can learn that neighbouring hours behave
alike instead of treating all 24 as unrelated. Both encodings were trained and the choice was made
on validation (see `reports/model_selection.md`).

## Features

```
categorical: hour, day_of_week, season
numeric:     temperature, humidity, wind_speed, visibility, dew_point,
             solar_radiation, rainfall, snowfall, is_holiday
```

`month` is deliberately excluded. Under a chronological split the training window covers only
months 12 and 1–7, so months 8–11 would reach the model as levels that never received a gradient.
Removing it improved validation MAE from 219.1 to 168.7.

No lagged target features. Adding them would force the API to fetch rental history before every
prediction and would complicate the delayed-target feedback loop planned for stage 5.

## Metrics

Primary metric: **WAPE** (`Σ|y−ŷ| / Σy`). MAE is reported alongside it and is what early stopping
and model selection optimise. MAPE is not used: hours with very low demand make it explode.

Business metric **WCE**: `mean(3·max(0, y−ŷ) + 1·max(0, ŷ−y))` — underforecasting leaves riders
without bikes and is weighted three times heavier than overforecasting.

| Model | val MAE | val WAPE | test MAE | test WAPE | test WCE |
| --- | --- | --- | --- | --- | --- |
| seasonal_median (baseline) | 548.8 | 0.566 | 422.9 | 0.497 | 1212.6 |
| hgb (reference) | 158.1 | 0.163 | 279.4 | 0.329 | 817.4 |
| mlp_onehot | 217.1 | 0.224 | 201.6 | 0.237 | 537.6 |
| **mlp_embedding (serving)** | 168.7 | 0.174 | **180.3** | **0.212** | **332.9** |

Test R² = 0.805. The serving model improves test MAE by 57 % over the seasonal baseline.

MAE is not comparable across splits: mean demand is 645 in train, 969 in validation and 850 in
test. Compare WAPE instead.

Gradient boosting wins on validation but loses on test — it overfits harder (train MAE 43 against
88) and transfers worse across the seasonal shift.

## Known limitations

- **Peaks are underforecast.** Worst hours are 08:00 (MAE 462, mean error −349) and 18:00
  (MAE 322, −254). Autumn peaks exceed summer ones and autumn is absent from training. This is the
  most expensive error under the business metric.
- **Rain.** On the 99 rainy test hours WAPE is 0.73 against 0.20 in dry weather and R² drops to
  0.25; demand is systematically overforecast.
- **Autumn is out of distribution.** The dataset is exactly one year, so autumn falls entirely in
  the test window. Unseen category levels are zeroed after training rather than left at random
  initialisation, so behaviour is deterministic, but autumn predictions rest on weather rather than
  on the season label.
- **No trend term.** At equal temperature, autumn demand is markedly higher than spring demand —
  the service grew over the year. The model cannot see that trend and therefore underforecasts
  late-period demand.
- **One city, no stations.** The dataset is a single aggregate counter for Seoul. The model cannot
  say anything about individual docks.
- **Weekends predict better than weekdays** (MAE 143 against 194).

## Intended use and misuse

Intended for operational planning of redistribution at city scale, on the horizon of the next hour.

Not suitable for per-station planning, for horizons beyond a few hours, for cities other than
Seoul, or for any decision about people. The dataset contains no personal data.

## Reproducibility

```bash
make install-ml
make data
make train
```

A clean rerun with `data/` and `models/` deleted reproduces every metric bit for bit: seeds are
fixed for Python, NumPy and torch, the DataLoader uses a seeded generator, and deterministic
algorithms are enabled.

Artifact metadata records `model_version`, the dataset sha256, the training window, the seed and
the versions of Python, torch and scikit-learn.

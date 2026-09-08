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
alike instead of treating all 24 as unrelated.

## How the champion is chosen

Every trained model is a candidate — baseline, gradient boosting and both networks. The choice is
made by **rolling-origin cross-validation** over the train and validation period; the test split is
never read, and a fold that overlaps it raises an error.

| Model | fold 1 (Apr–May) | fold 2 (Jun–Jul) | fold 3 (Aug–Sep) | mean MAE | worst fold |
| --- | --- | --- | --- | --- | --- |
| **mlp_embedding** | 407.2 | 378.0 | 379.2 | **388.1** | **407.2** |
| hgb | 579.6 | 469.5 | 200.2 | 416.4 | 579.6 |
| mlp_onehot | 480.2 | 358.8 | 439.3 | 426.1 | 480.2 |
| seasonal_median | 689.9 | 867.3 | 644.1 | 733.8 | 867.3 |

Fold MAE is higher than the headline numbers because each fold trains on a fraction of the data;
only the comparison between rows matters.

This is why a single window is not enough. Gradient boosting wins fold 3 outright, and fold 3 is
exactly the August–September window used as the validation split — which is why HGB has the better
validation MAE (158.1 against 168.7). On the two earlier folds it is the weakest of the three, and
its spread across folds is 200–580 against 378–407 for the network. Averaging over folds picks the
model that behaves consistently rather than the one that suits one season.

Reproduce with `python -m bikeflow.ml cv`; results are written to `reports/cv_folds.csv` and
`reports/cv_summary.csv`.

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

Primary metric: **MAE**, in rentals per hour, which is what early stopping and model selection
optimise. **WAPE** (`Σ|y−ŷ| / Σy`) is reported alongside it.

Within one window the two are the same ranking: `WAPE = MAE / mean(y)`, and the mean is a constant
there, so no model can win on one and lose on the other. WAPE earns its place when windows are
compared with each other — mean demand is 645 in train, 969 in validation and 850 in test, so raw
MAE is not comparable across them, and rolling quality tracking in stage 6 needs the scale-free
form. MAPE is not used: hours with very low demand make it explode.

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

Gradient boosting wins the single validation window but is the least consistent model across the
selection folds above, and it overfits harder (train MAE 43 against 88). The test column is
reported for completeness only; it played no part in choosing the champion.

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

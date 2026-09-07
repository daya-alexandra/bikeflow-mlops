# Models

No trained model is committed to Git. Artifacts here are produced locally by `make train`.

| File | Role |
| --- | --- |
| `model.joblib` | serving artifact — a copy of the winning model, loaded by `BikeflowPredictor` |
| `mlp_embedding.joblib` | PyTorch MLP with entity embeddings (currently the winner) |
| `mlp_onehot.joblib` | same network, one-hot categoricals — the runner-up |
| `hgb.joblib` | gradient boosting reference, and the Challenger for the stage 7 quality gate |
| `seasonal_median.joblib` | baseline: median demand per (hour, weekday) |

## Artifact format

One model is one joblib file with a fixed layout, so a consumer can load any of them without
knowing whether torch or scikit-learn is inside. `bikeflow.ml.models.registry` validates the layout
on load and refuses an artifact whose feature contract no longer matches the code.

```python
{
  "kind": "mlp_embedding" | "mlp_onehot" | "hgb" | "seasonal_median",
  "model": <estimator; torch models serialise as a state_dict>,
  "feature_spec": {"columns", "categorical", "numeric", "categories", "scaler"},
  "target": "rented_bike_count",
  "metadata": {"model_version", "trained_at", "data_sha256", "train_period",
               "seed", "python", "torch", "sklearn", "git_commit"},
  "metrics": {"train": {...}, "validation": {...}, "test": {...}},
}
```

Preprocessing travels with the estimator: feature order, category levels and the scaler statistics
all live in `feature_spec`, so online and offline transformations cannot diverge.

`model_version` is immutable and identifies what was trained, on which dataset, and when — for
example `mlp_embedding-373339b7-20260907T181012Z`.

See [`docs/model/model_card.md`](../docs/model/model_card.md) for architecture, metrics and
limitations.

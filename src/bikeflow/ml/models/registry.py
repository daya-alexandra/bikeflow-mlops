"""Model artifact format.

One model is one joblib file with a fixed dictionary layout, so the consumer
(the FastAPI service from stage 4 on) can load any of them without knowing
whether torch or scikit-learn is inside. Changing these keys is a breaking
change to the contract with role B.
"""

from __future__ import annotations

import datetime as dt
import platform
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..config import ensure_dir, load_config, resolve
from ..features import (
    CATEGORY_LEVELS,
    TARGET,
    categorical_columns,
    feature_columns,
    numeric_columns,
)

BUNDLE_KEYS = ("kind", "model", "feature_spec", "target", "metadata", "metrics")


def _versions() -> dict[str, str]:
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }


def feature_spec(model: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "columns": feature_columns(),
        "categorical": categorical_columns(),
        "numeric": numeric_columns(),
        "categories": {k: list(v) for k, v in CATEGORY_LEVELS.items()},
        "scaler": None,
    }
    mean = getattr(model, "mean_", None)
    std = getattr(model, "std_", None)
    if mean is not None and std is not None:
        spec["scaler"] = {
            "columns": numeric_columns(),
            "mean": np.asarray(mean).tolist(),
            "std": np.asarray(std).tolist(),
        }
    return spec


def save_bundle(
    path: str | Path,
    model: Any,
    metrics: dict[str, Any],
    data_sha256: str | None = None,
    train_period: tuple[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a model artifact. Returns the path written."""
    destination = resolve(path)
    ensure_dir(destination.parent)

    kind = getattr(model, "kind", type(model).__name__)
    trained_at = dt.datetime.now().astimezone()
    # Immutable identifier the API reports with every prediction: what was
    # trained, on which dataset, and when.
    model_version = (
        f"{kind}-{(data_sha256 or 'nodata')[:8]}-{trained_at.astimezone(dt.UTC):%Y%m%dT%H%M%SZ}"
    )

    bundle = {
        "kind": kind,
        "model": model,
        "feature_spec": feature_spec(model),
        "target": TARGET,
        "metadata": {
            "model_version": model_version,
            "trained_at": trained_at.isoformat(timespec="seconds"),
            "data_sha256": data_sha256,
            "train_period": list(train_period) if train_period else None,
            "seed": load_config()["seed"],
            "git_commit": None,
            **_versions(),
            **(extra or {}),
        },
        "metrics": metrics,
    }
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Read a model artifact and check it still matches the contract."""
    source = resolve(path)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run `python -m bikeflow train` first.")
    bundle = joblib.load(source)

    missing = [key for key in BUNDLE_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"{source} is not a BikeFlow artifact; missing keys {missing}")

    stored = bundle["feature_spec"]["columns"]
    expected = feature_columns()
    if stored != expected:
        raise ValueError(
            "Feature contract mismatch between artifact and current code.\n"
            f"  artifact: {stored}\n  code:     {expected}\n"
            "Retrain the model or restore the previous params.yaml."
        )
    return bundle


def promote(source: str | Path, destination: str | Path | None = None) -> Path:
    """Copy the winning model to the production artifact path."""
    target = resolve(destination or load_config()["paths"]["production_model"])
    ensure_dir(target.parent)
    shutil.copyfile(resolve(source), target)
    return target

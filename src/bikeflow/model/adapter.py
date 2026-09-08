"""Adapter between the API's `Predictor` protocol and the trained ML artifact.

This is the only place where the API contract meets the training pipeline. It
translates the request field names used by the service into the canonical
feature names used during training, then defers every transformation to
`bikeflow.ml.features` — the same module the training run used, so online and
offline preprocessing cannot drift apart.

Training, drift detection and retraining stay outside this boundary, as the
model/API contract requires.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from bikeflow.model.protocol import FeatureValue

#: API request field -> canonical training feature name.
FIELD_ALIASES: dict[str, str] = {
    "temperature_c": "temperature",
    "humidity_pct": "humidity",
    "wind_speed_m_s": "wind_speed",
    "visibility_10m": "visibility",
    "dew_point_c": "dew_point",
    "solar_radiation_mj_m2": "solar_radiation",
    "rainfall_mm": "rainfall",
    "snowfall_cm": "snowfall",
    "holiday": "is_holiday",
    "functioning_day": "is_functioning",
}

#: Fields whose names already match the training pipeline.
PASSTHROUGH: frozenset[str] = frozenset({"hour", "season", "day_of_week", "date"})


def to_canonical_row(features: Mapping[str, FeatureValue]) -> dict[str, FeatureValue]:
    """Rename API fields to the names the trained model expects."""
    row: dict[str, FeatureValue] = {}
    unknown: list[str] = []

    for key, value in features.items():
        if key in FIELD_ALIASES:
            row[FIELD_ALIASES[key]] = value
        elif key in PASSTHROUGH:
            row[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"Unknown feature field(s) for the trained model: {sorted(unknown)}. "
            f"Known: {sorted(set(FIELD_ALIASES) | set(PASSTHROUGH))}"
        )
    return row


class BikeflowPredictor:
    """Serve predictions from a trained BikeFlow artifact.

    Implements the `Predictor` protocol: one feature mapping in, one
    non-negative float out, plus an immutable artifact version.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        # Imported lazily so the API image does not pay for torch/pandas at
        # import time when it is running the stub instead.
        from bikeflow.ml.inference import Predictor as MLPredictor

        self._predictor = MLPredictor.load(model_path)
        self._model_version = str(self._predictor.metadata["model_version"])

    @property
    def model_version(self) -> str:
        """Identify the exact inference artifact."""
        return self._model_version

    def predict(self, features: Mapping[str, FeatureValue]) -> float:
        """Return a non-negative demand prediction for one feature row."""
        prediction = self._predictor.predict(to_canonical_row(features))
        return float(prediction[0])

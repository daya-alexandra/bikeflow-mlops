"""Create a tiny real HGB artifact for Docker/CI runtime smoke tests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from bikeflow.ml.features import build_features, coerce_input
from bikeflow.ml.models.hgb import HGBModel
from bikeflow.ml.models.registry import save_bundle


def create(destination: Path) -> Path:
    rows = []
    for index in range(96):
        hour = index % 24
        rows.append(
            {
                "date": (pd.Timestamp("2018-07-01") + pd.Timedelta(hours=index)).date().isoformat(),
                "hour": hour,
                "temperature": 18 + 8 * np.sin(hour / 24 * 2 * np.pi),
                "humidity": 55 + index % 20,
                "wind_speed": 1 + (index % 5) / 10,
                "visibility": 1800,
                "dew_point": 10,
                "solar_radiation": max(0.0, np.sin((hour - 6) / 12 * np.pi)),
                "rainfall": 1 if index % 19 == 0 else 0,
                "snowfall": 0,
                "season": "Summer",
                "is_holiday": False,
                "is_functioning": True,
            }
        )
    frame = pd.DataFrame(rows)
    data_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    features = build_features(coerce_input(frame))
    target = 180 + 12 * features["hour"].to_numpy() + 2 * frame["temperature"].to_numpy()
    params = {
        "loss": "poisson",
        "max_iter": 25,
        "learning_rate": 0.1,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "n_iter_no_change": 25,
    }
    model = HGBModel(params=params, seed=42).fit(features, target)
    return save_bundle(
        destination,
        model,
        metrics={"purpose": "runtime-smoke"},
        data_sha256=data_hash,
        training_params=params,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(create(args.destination))


if __name__ == "__main__":
    main()

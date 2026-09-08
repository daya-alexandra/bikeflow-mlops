"""Train candidates, select on validation, then evaluate the winner on test."""

from __future__ import annotations

import datetime as dt
import json
import random
from typing import Any

import numpy as np
import pandas as pd

from ..config import ensure_dir, load_config, resolve
from ..data.download import data_sha256
from ..data.split import SPLIT_NAMES, load_splits
from ..features import TARGET, build_features
from ..metrics import evaluate, evaluate_by_slice, rain_flag
from ..models.baseline import SeasonalMedianBaseline
from ..models.hgb import HGBModel
from ..models.registry import promote, save_bundle

MLP_KINDS = ("mlp_onehot", "mlp_embedding")
SERVING_MODEL = "hgb"


def seed_everything(seed: int) -> None:
    """Seed randomness used by the default scikit-learn training path."""

    random.seed(seed)
    np.random.seed(seed)


def prepare(parts: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Drop closed hours and build the feature matrix for each split."""

    prepared = {}
    for name in SPLIT_NAMES:
        raw = parts[name]
        working = raw[raw["is_functioning"]].reset_index(drop=True)
        prepared[name] = {
            "raw": working,
            "X": build_features(working),
            "y": working[TARGET].to_numpy(dtype="float64"),
        }
        dropped = len(raw) - len(working)
        print(f"[prepare] {name}: {len(working)} rows ({dropped} closed hours dropped)")
    return prepared


def fit_models(data: dict[str, dict[str, Any]], include_mlp: bool = False) -> dict[str, Any]:
    """Fit the baseline and HGB, optionally retaining the MLP experiment."""

    train, validation = data["train"], data["validation"]
    models: dict[str, Any] = {}
    print("\n[train] seasonal median baseline")
    models["seasonal_median"] = SeasonalMedianBaseline().fit(train["X"], train["y"])
    print("[train] HistGradientBoosting serving candidate")
    models["hgb"] = HGBModel().fit(train["X"], train["y"], validation["X"], validation["y"])

    if include_mlp:
        from ..models.torch_mlp import TorchMLPRegressor

        for encoding in load_config()["models"]["mlp"]["encodings"]:
            print(f"[train] optional torch MLP experiment ({encoding})")
            model = TorchMLPRegressor(encoding=encoding)
            model.fit(train["X"], train["y"], validation["X"], validation["y"])
            models[model.kind] = model
    return models


def score_selection(
    models: dict[str, Any], data: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Score candidates without touching the held-out test split."""

    return {
        name: {
            split: evaluate(data[split]["y"], model.predict(data[split]["X"]))
            for split in ("train", "validation")
        }
        for name, model in models.items()
    }


def choose_main(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Choose by validation MAE only; WAPE is reported as a secondary metric."""

    candidates = [name for name in results if name != "seasonal_median"]
    winner = min(candidates, key=lambda name: results[name]["validation"]["mae"])
    if winner != SERVING_MODEL:
        raise RuntimeError(
            f"Validation selected {winner}, but the provisional serving decision "
            f"is {SERVING_MODEL}. "
            "Review the team decision before promoting a different model."
        )
    return winner


def add_final_test(
    results: dict[str, dict[str, dict[str, float]]],
    model: Any,
    main_kind: str,
    data: dict[str, dict[str, Any]],
) -> None:
    """Evaluate only the already-selected model on the held-out test split."""

    test = data["test"]
    results[main_kind]["test"] = evaluate(test["y"], model.predict(test["X"]))


def slice_reports(
    models: dict[str, Any], data: dict[str, dict[str, Any]], main_kind: str
) -> dict[str, pd.DataFrame]:
    """Create slices while keeping test hidden from non-selected models."""

    by_season, by_hour = [], []
    for split in SPLIT_NAMES:
        names = [main_kind] if split == "test" else list(models)
        frame = data[split]["raw"].copy()
        frame["is_rain"] = rain_flag(frame)
        actual = data[split]["y"]
        for name in names:
            predicted = models[name].predict(data[split]["X"])
            for column, destination in (("season", by_season), ("hour", by_hour)):
                report = evaluate_by_slice(frame, actual, predicted, column)
                report.insert(0, "model", name)
                report.insert(0, "split", split)
                destination.append(report)
    return {
        "season": pd.concat(by_season, ignore_index=True),
        "hour": pd.concat(by_hour, ignore_index=True),
    }


def summary_table(results: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for model, per_split in results.items():
        row: dict[str, Any] = {"model": model}
        for split in SPLIT_NAMES:
            metrics = per_split.get(split)
            row[f"{split}_mae"] = round(metrics["mae"], 3) if metrics else "—"
            row[f"{split}_wape"] = round(metrics["wape"], 6) if metrics else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def write_reports(
    results: dict[str, dict[str, dict[str, float]]],
    slices: dict[str, pd.DataFrame],
    main_kind: str,
    data: dict[str, dict[str, Any]],
) -> None:
    reports = ensure_dir(load_config()["paths"]["reports_dir"])
    cfg = load_config()
    payload = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_sha256": data_sha256(),
        "seed": cfg["seed"],
        "selection_metric": "validation_mae",
        "secondary_metric": "wape",
        "test_policy": "evaluated only after selection and only for the selected model",
        "split": {name: cfg["split"][name] for name in SPLIT_NAMES},
        "split_sizes": {name: int(len(data[name]["raw"])) for name in SPLIT_NAMES},
        "main_model": main_kind,
        "models": results,
    }
    (reports / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    slices["season"].to_csv(reports / "metrics_by_season.csv", index=False)
    slices["hour"].to_csv(reports / "metrics_by_hour.csv", index=False)

    validation = results[main_kind]["validation"]
    test = results[main_kind]["test"]
    lines = [
        "# Выбор основной модели",
        "",
        f"Сгенерировано: {payload['generated_at']}",
        "",
        "Текущее решение команды — **HistGradientBoosting (`hgb`)**. Модель выбирается",
        "только по **MAE на validation**; WAPE служит дополнительной метрикой.",
        "Test не участвовал в выборе: он был рассчитан один раз после фиксации победителя",
        "и только для HGB.",
        "",
        f"- validation MAE: **{validation['mae']:.3f}**",
        f"- validation WAPE: **{validation['wape']:.6f}**",
        f"- test MAE: **{test['mae']:.3f}**",
        f"- test WAPE: **{test['wape']:.6f}**",
        "",
        "MLP сохранена как необязательный эксперимент (`--include-mlp`) и не нужна",
        "serving-контейнеру. Если validation выберет не HGB, pipeline остановится и потребует",
        "нового командного решения вместо молчаливой смены production-модели.",
        "",
        "## Метрики текущего запуска",
        "",
        summary_table(results).to_markdown(index=False),
        "",
        "Test состоит из последних двух месяцев годового набора; сезонный сдвиг — известное",
        "ограничение данных, а не причина использовать test для настройки модели.",
    ]
    (reports / "model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(model: Any, data: dict[str, dict[str, Any]], main_kind: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = ensure_dir(resolve(load_config()["paths"]["reports_dir"]) / "figures")
    test = data["test"]
    predicted = model.predict(test["X"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    window = test["raw"].head(24 * 14)
    axes[0].plot(window["timestamp"], test["y"][: len(window)], label="факт", linewidth=1.2)
    axes[0].plot(window["timestamp"], predicted[: len(window)], label="прогноз", linewidth=1.2)
    axes[0].set_title(f"Test, первые 14 дней — {main_kind}")
    axes[0].set_ylabel("аренд в час")
    axes[0].legend()
    axes[1].scatter(test["y"], predicted, s=6, alpha=0.35)
    limit = max(test["y"].max(), predicted.max())
    axes[1].plot([0, limit], [0, limit], color="black", linewidth=1)
    axes[1].set_xlabel("факт")
    axes[1].set_ylabel("прогноз")
    axes[1].set_title("Прогноз против факта, test")
    fig.savefig(figures / "pred_vs_actual_test.png", dpi=130)
    plt.close(fig)


def run_training(
    save: bool = True, figures: bool = True, include_mlp: bool = False
) -> dict[str, Any]:
    cfg = load_config()
    seed_everything(int(cfg["seed"]))
    data = prepare(load_splits())
    models = fit_models(data, include_mlp=include_mlp)
    print("\n[evaluate] selection stage: train + validation only")
    results = score_selection(models, data)
    main_kind = choose_main(results)
    print(f"[select] main model by validation MAE: {main_kind}")
    print("[evaluate] final stage: selected model on held-out test")
    add_final_test(results, models[main_kind], main_kind, data)
    slices = slice_reports(models, data, main_kind)
    print("\n" + summary_table(results).to_string(index=False))

    if save:
        models_dir = ensure_dir(cfg["paths"]["models_dir"])
        period = (cfg["split"]["train"][0], cfg["split"]["train"][1])
        for name, model in models.items():
            extra = {}
            if hasattr(model, "n_parameters"):
                extra = {"n_parameters": model.n_parameters(), "best_epoch": model.best_epoch_}
            save_bundle(
                models_dir / f"{name}.joblib",
                model,
                metrics=results[name],
                data_sha256=data_sha256(),
                train_period=period,
                training_params=getattr(model, "params", cfg["models"].get(name, {})),
                extra=extra,
            )
        production = promote(models_dir / f"{main_kind}.joblib")
        print(f"[save] production artifact -> {production}")
        write_reports(results, slices, main_kind, data)
        if figures:
            make_figures(models[main_kind], data, main_kind)

    return {"models": models, "results": results, "main": main_kind, "slices": slices}

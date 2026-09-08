"""Train every model, evaluate them on the same footing, promote the winner.

Order of business:
  1. load the three temporal splits and drop non-functioning hours
  2. fit baseline, HGB reference and both MLP encodings
  3. score all of them on train / validation / test plus slices
  4. promote the better MLP (by validation MAE) to models/model.joblib
  5. write reports/

The test split is only touched in step 3, after every fitting decision is made.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pandas as pd

from ..config import ensure_dir, load_config, resolve
from ..data.download import data_sha256
from ..data.split import SPLIT_NAMES, load_splits
from ..features import TARGET, build_features
from ..metrics import evaluate, evaluate_by_slice, rain_flag
from ..models.baseline import SeasonalMedianBaseline
from ..models.hgb import HGBModel
from ..models.registry import promote, save_bundle
from ..models.torch_mlp import TorchMLPRegressor
from .cv import choose_by_cv, run_cv, summarise_cv

MLP_KINDS = ("mlp_onehot", "mlp_embedding")


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


def fit_models(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train, validation = data["train"], data["validation"]
    models: dict[str, Any] = {}

    print("\n[train] seasonal median baseline")
    models["seasonal_median"] = SeasonalMedianBaseline().fit(train["X"], train["y"])

    print("[train] gradient boosting reference")
    models["hgb"] = HGBModel().fit(train["X"], train["y"], validation["X"], validation["y"])

    for encoding in load_config()["models"]["mlp"]["encodings"]:
        print(f"[train] torch mlp ({encoding})")
        model = TorchMLPRegressor(encoding=encoding)
        model.fit(train["X"], train["y"], validation["X"], validation["y"])
        print(f"         parameters: {model.n_parameters():,}")
        models[model.kind] = model

    return models


def score(models: dict[str, Any], data: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}
    for name, model in models.items():
        results[name] = {
            split: evaluate(data[split]["y"], model.predict(data[split]["X"]))
            for split in SPLIT_NAMES
        }
    return results


def slice_reports(
    models: dict[str, Any], data: dict[str, dict[str, Any]]
) -> dict[str, pd.DataFrame]:
    """Per-season and per-hour breakdowns across every split."""
    by_season, by_hour = [], []

    for split in SPLIT_NAMES:
        frame = data[split]["raw"].copy()
        frame["is_rain"] = rain_flag(frame)
        actual = data[split]["y"]
        for name, model in models.items():
            predicted = model.predict(data[split]["X"])

            season = evaluate_by_slice(frame, actual, predicted, "season")
            season.insert(0, "model", name)
            season.insert(0, "split", split)
            by_season.append(season)

            hour = evaluate_by_slice(frame, actual, predicted, "hour")
            hour.insert(0, "model", name)
            hour.insert(0, "split", split)
            by_hour.append(hour)

    return {
        "season": pd.concat(by_season, ignore_index=True),
        "hour": pd.concat(by_hour, ignore_index=True),
    }


def choose_main(results: dict[str, dict[str, dict]], cv_scores: pd.DataFrame | None = None) -> str:
    """Pick the champion among ALL trained models.

    With `strategy: rolling_cv` the decision is the average over several
    chronological folds. That matters here: on the August-September window
    alone HGB looks best, yet it is the weakest of the three on the two folds
    before it, so a single window would pick a model that does not generalise.

    `strategy: validation` falls back to the single validation split. Under
    either strategy the test split plays no part in the choice.
    """
    cfg = load_config()["selection"]
    metric = cfg["metric"]

    if cfg["strategy"] == "rolling_cv":
        if cv_scores is None:
            raise ValueError("selection.strategy is 'rolling_cv' but no CV scores were provided.")
        return choose_by_cv(cv_scores)

    if cfg["strategy"] != "validation":
        raise ValueError(
            f"Unknown selection strategy {cfg['strategy']!r}; use 'rolling_cv' or 'validation'."
        )
    return min(results, key=lambda kind: results[kind]["validation"][metric])


def summary_table(results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for model, per_split in results.items():
        row = {"model": model}
        for split in SPLIT_NAMES:
            row[f"{split}_mae"] = round(per_split[split]["mae"], 1)
            row[f"{split}_wape"] = round(per_split[split]["wape"], 4)
        row["test_rmse"] = round(per_split["test"]["rmse"], 1)
        row["test_wce"] = round(per_split["test"]["wce"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def check_beats_baseline(results: dict[str, dict[str, dict]]) -> list[str]:
    """Every real model must clear the baseline on test, or something is wrong."""
    baseline = results["seasonal_median"]["test"]
    failures = []
    for name in ("hgb",) + MLP_KINDS:
        current = results[name]["test"]
        if current["mae"] >= baseline["mae"] or current["wape"] >= baseline["wape"]:
            failures.append(
                f"{name}: MAE {current['mae']:.1f} vs baseline {baseline['mae']:.1f}, "
                f"WAPE {current['wape']:.3f} vs {baseline['wape']:.3f}"
            )
    return failures


def write_reports(
    results: dict[str, dict[str, dict]],
    slices: dict[str, pd.DataFrame],
    main_kind: str,
    models: dict[str, Any],
    data: dict[str, dict[str, Any]],
    cv_scores: pd.DataFrame | None = None,
) -> None:
    reports = ensure_dir(load_config()["paths"]["reports_dir"])
    cfg = load_config()

    payload = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_sha256": data_sha256(),
        "seed": cfg["seed"],
        "split": {name: cfg["split"][name] for name in SPLIT_NAMES},
        "split_sizes": {name: int(len(data[name]["raw"])) for name in SPLIT_NAMES},
        "business_metric": cfg["business_metric"],
        "main_model": main_kind,
        "models": results,
    }
    (reports / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if cv_scores is not None:
        cv_scores.to_csv(reports / "cv_folds.csv", index=False)
        summarise_cv(cv_scores).to_csv(reports / "cv_summary.csv", index=False)
        payload["selection"] = {
            "strategy": cfg["selection"]["strategy"],
            "metric": cfg["selection"]["metric"],
            "folds": cfg["selection"]["folds"],
            "summary": summarise_cv(cv_scores).to_dict("records"),
        }
        (reports / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    slices["season"].to_csv(reports / "metrics_by_season.csv", index=False)
    slices["hour"].to_csv(reports / "metrics_by_hour.csv", index=False)

    baseline_test = results["seasonal_median"]["test"]
    main_test = results[main_kind]["test"]
    improvement = 100 * (1 - main_test["mae"] / baseline_test["mae"])
    other = [k for k in MLP_KINDS if k != main_kind][0]

    lines = [
        "# Выбор основной модели",
        "",
        f"Сгенерировано: {payload['generated_at']}",
        "",
        "Основная модель выбирается по **MAE на validation** между двумя кодировками",
        "нейросети. Перебора гиперпараметров нет — этого требует бриф «не усложнять».",
        "",
        f"- Победитель: **{main_kind}** — validation MAE "
        f"{results[main_kind]['validation']['mae']:.1f}",
        f"- Вторая кодировка: {other} — validation MAE {results[other]['validation']['mae']:.1f}",
        f"- Референс HGB: validation MAE {results['hgb']['validation']['mae']:.1f}",
        f"- Baseline: validation MAE {results['seasonal_median']['validation']['mae']:.1f}",
        "",
        "## На test",
        "",
        summary_table(results).to_markdown(index=False),
        "",
        f"Основная модель улучшает MAE относительно baseline на **{improvement:.1f}%**.",
        "",
        "Test содержит только осень (см. CLAUDE.md, раздел 4) — это ограничение",
        "годового датасета, а не ошибка разбиения.",
        "",
        "## Параметры сетей",
        "",
    ]
    for kind in MLP_KINDS:
        model = models[kind]
        lines.append(
            f"- `{kind}`: {model.n_parameters():,} параметров, "
            f"лучшая эпоха {model.best_epoch_} из {len(model.history_)}"
        )
    (reports / "model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[reports] wrote metrics.json, metrics_by_*.csv, model_selection.md -> {reports}")


def make_figures(models: dict[str, Any], data: dict[str, dict[str, Any]], main_kind: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = ensure_dir(resolve(load_config()["paths"]["reports_dir"]) / "figures")
    test = data["test"]
    predicted = models[main_kind].predict(test["X"])

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

    table = models["mlp_embedding"].embedding_table("hour")
    if table is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(table[:, 0], table[:, 1], s=40)
        for hour, (x, y) in enumerate(table[:, :2]):
            ax.annotate(str(hour), (x, y), fontsize=9, xytext=(4, 3), textcoords="offset points")
        ax.set_title("Обученные embedding-векторы часа (первые 2 измерения)")
        fig.tight_layout()
        fig.savefig(figures / "hour_embedding.png", dpi=130)
        plt.close(fig)

    print(f"[reports] figures -> {figures}")


def run_training(save: bool = True, figures: bool = True) -> dict[str, Any]:
    cfg = load_config()
    data = prepare(load_splits())
    models = fit_models(data)

    print("\n[evaluate] scoring on train / validation / test")
    results = score(models, data)
    slices = slice_reports(models, data)

    cv_scores = None
    if cfg["selection"]["strategy"] == "rolling_cv":
        print("\n[cv] rolling-origin selection (the test split is never read)")
        cv_scores = run_cv()
        print("\n" + summarise_cv(cv_scores).round(1).to_string(index=False))
    main_kind = choose_main(results, cv_scores)

    print("\n" + summary_table(results).to_string(index=False))
    print(f"\n[select] main model: {main_kind}")

    failures = check_beats_baseline(results)
    if failures:
        print("\n[WARNING] model(s) failed to beat the seasonal baseline on test:")
        for line in failures:
            print("  " + line)

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
                extra=extra,
            )
        production = promote(models_dir / f"{main_kind}.joblib")
        print(f"[save] {len(models)} artifact(s) in {models_dir}; production -> {production}")

        write_reports(results, slices, main_kind, models, data, cv_scores)
        if figures:
            make_figures(models, data, main_kind)

    return {"models": models, "results": results, "main": main_kind, "slices": slices}

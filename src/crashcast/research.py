"""Leakage-safe weekly police-beat collision-risk experiment.

The module intentionally uses only fields that are available before the
forecast week. It does not geocode block-level police reports or claim that
ADT counts are a valid exposure denominator without a spatial crosswalk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance,
)

FEATURES = [
    "beat_code", "week_sin", "week_cos", "lag_1", "lag_2", "lag_4", "lag_8",
    "lag_12", "lag_52", "roll_4", "roll_12", "roll_52", "injured_roll_12",
    "killed_roll_52", "hit_run_roll_12",
]


@dataclass(frozen=True)
class SourceSummary:
    path: str
    sha256: str
    records: int
    columns: list[str]
    missing_values: dict[str, int]
    date_min: str
    date_max: str
    unique_reports: int
    unique_beats: int


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_collisions(path: Path) -> tuple[pd.DataFrame, SourceSummary]:
    raw = pd.read_csv(path, dtype={"REPORT_ID": "string", "POLICE_BEAT": "string"})
    required = {"REPORT_ID", "DATE_TIME", "POLICE_BEAT", "INJURED", "KILLED", "HIT_RUN_LVL"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Collision source is missing required columns: {sorted(missing)}")
    frame = raw.copy()
    frame["date_time"] = pd.to_datetime(frame["DATE_TIME"], errors="coerce")
    frame["police_beat"] = frame["POLICE_BEAT"].str.strip()
    frame = frame.dropna(subset=["REPORT_ID", "date_time", "police_beat"])
    frame = frame.loc[frame["police_beat"].ne("")].drop_duplicates("REPORT_ID", keep="first")
    for field in ("INJURED", "KILLED"):
        frame[field.lower()] = pd.to_numeric(frame[field], errors="coerce").fillna(0).clip(lower=0)
    frame["hit_run"] = frame["HIT_RUN_LVL"].notna().astype(int)
    summary = SourceSummary(
        path=str(path), sha256=file_hash(path), records=int(len(frame)), columns=list(raw.columns),
        missing_values={key: int(value) for key, value in raw.isna().sum().items()},
        date_min=str(frame["date_time"].min()), date_max=str(frame["date_time"].max()),
        unique_reports=int(frame["REPORT_ID"].nunique()), unique_beats=int(frame["police_beat"].nunique()),
    )
    return frame, summary


def make_panel(collisions: pd.DataFrame) -> pd.DataFrame:
    events = collisions.copy()
    events["week"] = events["date_time"].dt.to_period("W-SUN").dt.start_time
    grouped = events.groupby(["police_beat", "week"], as_index=False).agg(
        collisions=("REPORT_ID", "size"), injured=("injured", "sum"),
        killed=("killed", "sum"), hit_run=("hit_run", "sum"),
    )
    # A complete beat-week panel is necessary: no report in a week means zero,
    # not missing. The final partial week is excluded below.
    weeks = pd.date_range(grouped["week"].min(), grouped["week"].max(), freq="W-MON")
    beats = sorted(grouped["police_beat"].unique())
    panel = pd.MultiIndex.from_product([beats, weeks], names=["police_beat", "week"]).to_frame(index=False)
    panel = panel.merge(grouped, how="left", on=["police_beat", "week"])
    for column in ("collisions", "injured", "killed", "hit_run"):
        panel[column] = panel[column].fillna(0).astype(float)
    panel = panel.sort_values(["police_beat", "week"]).reset_index(drop=True)
    # Do not evaluate a partial last week (source is updated daily).
    complete_through = panel["week"].max() - pd.Timedelta(days=7)
    return panel.loc[panel["week"] <= complete_through].copy()


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values(["police_beat", "week"])
    grouped = df.groupby("police_beat", group_keys=False)
    for lag in (1, 2, 4, 8, 12, 52):
        df[f"lag_{lag}"] = grouped["collisions"].shift(lag)
    for window in (4, 12, 52):
        df[f"roll_{window}"] = grouped["collisions"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    for source, output, window in (("injured", "injured_roll_12", 12), ("killed", "killed_roll_52", 52), ("hit_run", "hit_run_roll_12", 12)):
        df[output] = grouped[source].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    week_number = df["week"].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week_number / 52.18)
    df["week_cos"] = np.cos(2 * np.pi * week_number / 52.18)
    df["beat_code"] = pd.Categorical(df["police_beat"]).codes.astype(float)
    # Target is a future observation. Every feature above only sees weeks < t.
    df["next_week_count"] = grouped["collisions"].shift(-1)
    df["next_week_any"] = (df["next_week_count"] > 0).astype(float)
    return df.dropna(subset=FEATURES + ["next_week_count"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 2026 is intentionally held out of reported evaluation because the year is incomplete.
    train = df.loc[df["week"] < "2024-01-01"].copy()
    validate = df.loc[(df["week"] >= "2024-01-01") & (df["week"] < "2025-01-01")].copy()
    test = df.loc[(df["week"] >= "2025-01-01") & (df["week"] < "2026-01-01")].copy()
    if min(map(len, (train, validate, test))) == 0:
        raise ValueError("Temporal split is empty; inspect current source coverage.")
    return train, validate, test


def count_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    safe = np.clip(predicted, 1e-6, None)
    return {
        "mae": float(mean_absolute_error(observed, safe)),
        "rmse": float(mean_squared_error(observed, safe) ** 0.5),
        "poisson_deviance": float(mean_poisson_deviance(observed, safe)),
    }


def fit_and_evaluate(df: pd.DataFrame, output: Path) -> dict:
    train, validate, test = temporal_split(df)
    x_train, y_train = train[FEATURES], train["next_week_count"]
    x_valid, y_valid = validate[FEATURES], validate["next_week_count"]
    x_test, y_test = test[FEATURES], test["next_week_count"]
    # Seasonal-naive baseline uses the year-ago count if present; otherwise the
    # prior 12-week mean. Both values were available at prediction time.
    baseline_test = test["lag_52"].fillna(test["roll_12"]).to_numpy()
    candidates = {15: HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=15, l2_regularization=1.0, max_iter=250, learning_rate=0.06, random_state=42),
                  31: HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=31, l2_regularization=1.0, max_iter=250, learning_rate=0.06, random_state=42)}
    selected_leaves, selected = min(candidates.items(), key=lambda item: mean_absolute_error(y_valid, item[1].fit(x_train, y_train).predict(x_valid)))
    selected.fit(pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
    predicted = np.clip(selected.predict(x_test), 1e-6, None)
    classifier = HistGradientBoostingClassifier(max_leaf_nodes=15, l2_regularization=1.0, max_iter=180, learning_rate=0.06, random_state=42)
    classifier.fit(pd.concat([x_train, x_valid]), pd.concat([train["next_week_any"], validate["next_week_any"]]))
    probability = classifier.predict_proba(x_test)[:, 1]
    observed_binary = (y_test > 0).astype(int)
    test_output = test[["police_beat", "week", "next_week_count"]].rename(columns={"next_week_count": "observed_count"}).copy()
    test_output["baseline_count"] = baseline_test
    test_output["predicted_count"] = predicted
    test_output["predicted_any_probability"] = probability
    perm = permutation_importance(selected, x_test, y_test, scoring="neg_mean_absolute_error", n_repeats=12, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({"feature": FEATURES, "mae_importance_mean": perm.importances_mean, "mae_importance_std": perm.importances_std}).sort_values("mae_importance_mean", ascending=False)
    metrics = {
        "task": "next-week collision count and relative risk ranking by San Diego Police beat",
        "selection": {"validation_year": 2024, "selected_max_leaf_nodes": selected_leaves, "criterion": "lowest validation MAE"},
        "split_rows": {"train": len(train), "validation": len(validate), "test": len(test)},
        "test_period": {"start": str(test["week"].min().date()), "end": str(test["week"].max().date())},
        "count_models": {"seasonal_naive": count_metrics(y_test.to_numpy(), baseline_test), "hist_gradient_boosting_poisson": count_metrics(y_test.to_numpy(), predicted)},
        "occurrence_model": {"average_precision": float(average_precision_score(observed_binary, probability)), "brier_score": float(brier_score_loss(observed_binary, probability)), "positive_rate": float(observed_binary.mean())},
        "leakage_controls": ["chronological train/validation/test split", "lag and rolling features shifted by one week", "no coordinates inferred from block-level source", "2026 incomplete data excluded from reported test metrics"],
    }
    output.mkdir(parents=True, exist_ok=True)
    test_output.to_csv(output / "weekly_predictions.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    (output / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    joblib.dump({"count_model": selected, "occurrence_model": classifier, "features": FEATURES}, output / "models.joblib")
    return {"metrics": metrics, "predictions": test_output, "importance": importance}


def plot_outputs(panel: pd.DataFrame, result: dict, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    accent, muted, text = "#b7d9c0", "#829486", "#e7ece7"
    monthly = panel.assign(month=panel["week"].dt.to_period("M").dt.start_time).groupby("month")["collisions"].sum()
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(monthly.index, monthly.values, color=accent, linewidth=1.8)
    ax.set(title="Reported San Diego Police Collision Counts", ylabel="Reports per month", xlabel="Month")
    ax.grid(alpha=.18); ax.tick_params(colors=text)
    fig.savefig(figure_dir / "collision_history.png", dpi=220, facecolor="#111512"); plt.close(fig)
    pred = result["predictions"].copy(); aggregated = pred.groupby("week")[["observed_count", "predicted_count", "baseline_count"]].sum()
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(aggregated.index, aggregated.observed_count, label="Observed", color=text, linewidth=1.6)
    ax.plot(aggregated.index, aggregated.predicted_count, label="ML forecast", color=accent, linewidth=1.5)
    ax.plot(aggregated.index, aggregated.baseline_count, label="Seasonal-naive", color=muted, linewidth=1.2, linestyle="--")
    ax.set(title="Held-out Weekly Collision Counts (2025)", ylabel="Reports across beats", xlabel="Forecast week")
    ax.legend(frameon=False); ax.grid(alpha=.18)
    fig.savefig(figure_dir / "held_out_forecast.png", dpi=220, facecolor="#111512"); plt.close(fig)
    top = result["importance"].head(10).sort_values("mae_importance_mean")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.barh(top.feature, top.mae_importance_mean, xerr=top.mae_importance_std, color=accent, alpha=.9)
    ax.set(title="Permutation Feature Importance", xlabel="Increase in held-out MAE after permutation")
    ax.grid(axis="x", alpha=.18)
    fig.savefig(figure_dir / "feature_importance.png", dpi=220, facecolor="#111512"); plt.close(fig)


def run(raw_path: Path, root: Path) -> dict:
    collisions, source = load_collisions(raw_path)
    panel = make_panel(collisions)
    engineered = engineer_features(panel)
    processed = root / "data" / "processed"; processed.mkdir(parents=True, exist_ok=True)
    panel.to_csv(processed / "weekly_beat_panel.csv", index=False)
    engineered.to_csv(processed / "model_frame.csv", index=False)
    provenance = {"collision_source": asdict(source), "spatial_unit": "San Diego Police beat", "time_unit": "week beginning Monday", "coordinate_policy": "No collision coordinates are inferred or fabricated; source-level block address fields are not treated as point geometry."}
    (root / "data" / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result = fit_and_evaluate(engineered, root / "experiments" / "results")
    plot_outputs(panel, result, root / "reports" / "figures")
    return {"source": asdict(source), **result["metrics"]}

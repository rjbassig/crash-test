#!/usr/bin/env python3
"""Expanded, leakage-safe CRASH TEST evaluation using real source artifacts.

This intentionally reads the exact engineered frame produced by the baseline
pipeline and writes a *new* experiment directory, leaving historic results
unchanged. All validation is chronological; spatial holdout uses disjoint
police beats.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, mean_poisson_deviance

from crashcast.research import FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "expanded"
FIG = ROOT / "reports" / "expanded-figures"
RNG = np.random.default_rng(20261107)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, None)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(mean_squared_error(y, p) ** .5), "poisson_deviance": float(mean_poisson_deviance(y, p))}


def ranking(y: np.ndarray, score: np.ndarray, fraction: float = .1) -> dict[str, float]:
    n = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(score)[-n:]
    actual = np.argsort(y)[-n:]
    return {"top_fraction": fraction, "recall_at_top_k": float(len(set(chosen).intersection(actual)) / n), "lift": float(y[chosen].mean() / max(y.mean(), 1e-8))}


def bootstrap_mae_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, repeats: int = 800) -> dict[str, float]:
    deltas = []
    for _ in range(repeats):
        take = RNG.integers(0, len(y), len(y))
        deltas.append(mean_absolute_error(y[take], a[take]) - mean_absolute_error(y[take], b[take]))
    return {"mean_improvement": float(np.mean(deltas)), "ci_95_low": float(np.quantile(deltas, .025)), "ci_95_high": float(np.quantile(deltas, .975))}


def fit_count(train: pd.DataFrame, test: pd.DataFrame, feature_set: list[str]) -> tuple[dict[str, np.ndarray], HistGradientBoostingRegressor]:
    x, y = train[feature_set], train.next_week_count
    poisson = PoissonRegressor(alpha=.15, max_iter=600).fit(x, y)
    hgb = HistGradientBoostingRegressor(loss="poisson", max_iter=350, max_leaf_nodes=31, learning_rate=.05, l2_regularization=1, random_state=42).fit(x, y)
    return {"seasonal_naive": test.lag_52.fillna(test.roll_12).to_numpy(), "poisson_glm": poisson.predict(test[feature_set]), "hist_gradient_boosting": hgb.predict(test[feature_set])}, hgb


def evaluate_window(frame: pd.DataFrame, train_end: str, test_start: str, test_end: str) -> tuple[dict, pd.DataFrame, HistGradientBoostingRegressor]:
    train = frame[frame.week < train_end].copy(); test = frame[(frame.week >= test_start) & (frame.week < test_end)].copy()
    predictions, model = fit_count(train, test, FEATURES)
    output = test[["police_beat", "week", "next_week_count"]].rename(columns={"next_week_count": "observed_count"})
    record = {"train_end": train_end, "test_start": test_start, "test_end": test_end, "rows": len(test), "models": {}}
    for name, prediction in predictions.items():
        prediction = np.clip(prediction, 1e-6, None); output[name] = prediction
        record["models"][name] = {**metrics(output.observed_count.to_numpy(), prediction), **ranking(output.observed_count.to_numpy(), prediction)}
    return record, output, model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data" / "processed" / "model_frame.csv", parse_dates=["week"])
    # Two rolling origins whose test windows are wholly after their training cutoffs.
    rolling = []
    for args in [("2023-01-01", "2023-01-01", "2024-01-01"), ("2024-01-01", "2024-01-01", "2025-01-01")]:
        result, _, _ = evaluate_window(frame, *args); rolling.append(result)
    held, prediction, final_model = evaluate_window(frame, "2025-01-01", "2025-01-01", "2026-01-01")
    # Classification is trained on pre-2024 observations, calibrated on 2024,
    # then evaluated once on 2025. Calibration never sees the test year.
    train = frame[frame.week < "2024-01-01"]; calibration = frame[(frame.week >= "2024-01-01") & (frame.week < "2025-01-01")]; test = frame[(frame.week >= "2025-01-01") & (frame.week < "2026-01-01")]
    classifier = HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1, random_state=42).fit(train[FEATURES], train.next_week_any)
    raw_cal = classifier.predict_proba(calibration[FEATURES])[:, 1]
    calibrator = LogisticRegression(C=1e4).fit(raw_cal.reshape(-1, 1), calibration.next_week_any)
    raw_test = classifier.predict_proba(test[FEATURES])[:, 1]
    probability = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    observed_any = test.next_week_any.astype(int).to_numpy()
    occurrence = {"average_precision": float(average_precision_score(observed_any, probability)), "brier_score": float(brier_score_loss(observed_any, probability)), "base_rate": float(observed_any.mean()), "calibration_protocol": "Logistic calibration fit on 2024 only"}
    prediction["occurrence_probability"] = probability
    # Disjoint-beat generalization: selected deterministic 20% of beats are
    # never available during fitting, and only 2025 records score the test.
    beats = np.array(sorted(frame.police_beat.unique())); held_beats = set(beats[::5])
    spatial_train = frame[(frame.week < "2025-01-01") & ~frame.police_beat.isin(held_beats)]
    spatial_test = frame[(frame.week >= "2025-01-01") & (frame.week < "2026-01-01") & frame.police_beat.isin(held_beats)]
    spatial_pred, _ = fit_count(spatial_train, spatial_test, FEATURES)
    spatial = {"holdout_beats": len(held_beats), "test_rows": len(spatial_test), "hist_gradient_boosting": metrics(spatial_test.next_week_count.to_numpy(), spatial_pred["hist_gradient_boosting"])}
    # Ablation tests time-only vs. full history/severity features on exactly the
    # same train/test period, preserving comparability.
    time_only = ["beat_code", "week_sin", "week_cos", "lag_1", "lag_2", "lag_4", "lag_8", "lag_12", "lag_52", "roll_4", "roll_12", "roll_52"]
    _, ablated = fit_count(frame[frame.week < "2025-01-01"], test, time_only)
    ablated_pred = ablated.predict(test[time_only])
    ablation = {"time_history_only": metrics(test.next_week_count.to_numpy(), ablated_pred), "full_model": held["models"]["hist_gradient_boosting"]}
    interval = bootstrap_mae_delta(prediction.observed_count.to_numpy(), prediction.seasonal_naive.to_numpy(), prediction.hist_gradient_boosting.to_numpy())
    bins_true, bins_pred = calibration_curve(observed_any, probability, n_bins=10, strategy="quantile")
    calibration_rows = pd.DataFrame({"mean_predicted_probability": bins_pred, "observed_frequency": bins_true})
    # Per-beat error analysis uses rows from untouched 2025 predictions.
    by_beat = prediction.assign(abs_error=np.abs(prediction.observed_count - prediction.hist_gradient_boosting)).groupby("police_beat", as_index=False).agg(rows=("observed_count", "size"), observed_mean=("observed_count", "mean"), predicted_mean=("hist_gradient_boosting", "mean"), mae=("abs_error", "mean")).sort_values("mae", ascending=False)
    summary = {"experiment": "crash-test-expanded-v1", "task": "next-week collision count and relative risk ranking by San Diego Police beat", "historical_baseline_artifacts": "experiments/historical/baseline-v0", "held_out_2025": held, "rolling_origin": rolling, "occurrence_calibration": occurrence, "spatial_generalization": spatial, "ablation": ablation, "bootstrap_mae_improvement_over_seasonal_naive": interval, "leakage_controls": ["chronological split", "one-week shifted lags and rolling features", "2025 test year never used for model or calibrator fitting", "disjoint-beat spatial test", "no inferred point coordinates"]}
    prediction.to_csv(OUT / "held_out_predictions.csv", index=False); calibration_rows.to_csv(OUT / "calibration_curve.csv", index=False); by_beat.to_csv(OUT / "error_by_beat.csv", index=False); (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Figures are direct renderings of saved output tables.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained"); fig.patch.set_facecolor("#151919"); ax.set_facecolor("#151919"); names = list(held["models"]); values = [held["models"][name]["mae"] for name in names]; ax.bar(names, values, color=["#77847d", "#aabbb0", "#b7d9c0"]); ax.set(ylabel="MAE (collisions per beat-week)"); fig.savefig(FIG / "model_comparison.png", dpi=220, facecolor="#151919"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained"); fig.patch.set_facecolor("#151919"); ax.set_facecolor("#151919"); ax.plot([0, 1], [0, 1], "--", color="#728078", label="Perfect calibration"); ax.plot(bins_pred, bins_true, "o-", color="#b7d9c0", label="2025 calibrated classifier"); ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency"); ax.legend(); fig.savefig(FIG / "calibration.png", dpi=220, facecolor="#151919"); plt.close(fig)
    rank = prediction.groupby("police_beat", as_index=False).agg(observed=("observed_count", "mean"), predicted=("hist_gradient_boosting", "mean")).sort_values("predicted", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained"); fig.patch.set_facecolor("#151919"); ax.set_facecolor("#151919"); x=np.arange(len(rank)); ax.bar(x-.2, rank.observed, .4, label="Observed", color="#dfe5df"); ax.bar(x+.2, rank.predicted, .4, label="Predicted", color="#b7d9c0"); ax.set(xlabel="Police beat", ylabel="Mean weekly collision reports"); ax.set_xticks(x, rank.police_beat, rotation=45); ax.legend(); fig.savefig(FIG / "risk_ranking.png", dpi=220, facecolor="#151919"); plt.close(fig)


if __name__ == "__main__": main()

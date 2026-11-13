#!/usr/bin/env python3
"""Create small, public aggregate-only artifacts for the CRASH TEST dashboard."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "public"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predicted = pd.read_csv(ROOT / "experiments" / "expanded" / "held_out_predictions.csv", parse_dates=["week"])
    # Aggregate strictly by official administrative beat. No individual points or
    # inferred locations leave the research pipeline.
    beats = predicted.groupby("police_beat", as_index=False).agg(
        observed_mean=("observed_count", "mean"), predicted_mean=("hist_gradient_boosting", "mean"),
        baseline_mean=("seasonal_naive", "mean"), risk_rank_score=("hist_gradient_boosting", "mean"),
        occurrence_probability=("occurrence_probability", "mean"), weeks=("week", "size"),
    ).sort_values("risk_rank_score", ascending=False).reset_index(drop=True)
    beats["risk_rank"] = beats.index + 1
    beats["relative_risk"] = beats.risk_rank_score / beats.risk_rank_score.mean()
    beats["police_beat"] = beats.police_beat.astype(str)
    weekly = predicted.groupby("week", as_index=False).agg(observed=("observed_count", "sum"), predicted=("hist_gradient_boosting", "sum"), baseline=("seasonal_naive", "sum"))
    errors = pd.read_csv(ROOT / "experiments" / "expanded" / "error_by_beat.csv"); errors["police_beat"] = errors.police_beat.astype(str)
    calibration = pd.read_csv(ROOT / "experiments" / "expanded" / "calibration_curve.csv")
    importance = pd.read_csv(ROOT / "experiments" / "results" / "feature_importance.csv")
    summary = json.loads((ROOT / "experiments" / "expanded" / "summary.json").read_text())
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text())
    (OUT / "beats.json").write_text(beats.to_json(orient="records"))
    (OUT / "weekly_forecast.json").write_text(weekly.to_json(orient="records", date_format="iso"))
    (OUT / "error_by_beat.json").write_text(errors.to_json(orient="records"))
    (OUT / "calibration.json").write_text(calibration.to_json(orient="records"))
    (OUT / "importance.json").write_text(importance.to_json(orient="records"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    # Official topology is deliberately retained as topology; geometry is never
    # extrapolated from collision addresses.
    source = ROOT / "data" / "raw" / "sdpd_beats.topo.json"
    if not source.exists():
        raise FileNotFoundError("Missing official police-beat TopoJSON. Download it before building map artifacts.")
    shutil.copyfile(source, OUT / "sdpd_beats.topo.json")


if __name__ == "__main__":
    main()

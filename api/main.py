"""Serve only artifacts produced by `scripts/run_research.py`."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "expanded"
PROVENANCE = ROOT / "data" / "provenance.json"
PUBLIC = ROOT / "artifacts" / "public"
origins = [value.strip() for value in os.getenv("CRASHCAST_CORS_ORIGINS", "http://localhost:3000").split(",") if value.strip()]

app = FastAPI(title="CRASH TEST Research API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])


def require(path: Path) -> Path:
    if not path.exists():
        raise HTTPException(503, "Research artifacts are unavailable. Run scripts/run_research.py before serving the API.")
    return path


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts_ready": RESULTS.joinpath("summary.json").exists(), "data_provenance_ready": PROVENANCE.exists()}


@app.get("/api/overview")
def overview() -> dict:
    return {"summary": json.loads(require(RESULTS / "summary.json").read_text()), "provenance": json.loads(require(PROVENANCE).read_text())}


def public_json(name: str) -> list | dict:
    return json.loads(require(PUBLIC / name).read_text())


@app.get("/api/geography/beats")
def beat_geography() -> dict:
    """Official SDPD TopoJSON plus aggregate-only verified beat metrics."""
    return {"topology": public_json("sdpd_beats.topo.json"), "metrics": public_json("beats.json"), "source": "City of San Diego / SDPD law beats"}


@app.get("/api/history")
def history() -> list[dict]:
    return public_json("weekly_forecast.json")


@app.get("/api/forecasts/{beat}")
def beat_forecast(beat: str) -> list[dict]:
    frame = pd.read_csv(require(RESULTS / "held_out_predictions.csv"), parse_dates=["week"])
    result = frame.loc[frame["police_beat"].astype(str) == beat].copy()
    if result.empty:
        raise HTTPException(404, f"No verified held-out forecast for police beat {beat}.")
    return json.loads(result.to_json(orient="records", date_format="iso"))


@app.get("/api/diagnostics")
def diagnostics() -> dict:
    """Verified held-out records used by the dashboard's linked diagnostics."""
    predictions = pd.read_csv(require(RESULTS / "held_out_predictions.csv"), parse_dates=["week"])
    return {
        "predictions": json.loads(predictions.to_json(orient="records", date_format="iso")),
        "error_by_beat": public_json("error_by_beat.json"),
    }


@app.get("/api/models")
def models() -> dict:
    summary = public_json("summary.json")
    return {"held_out_2025": summary["held_out_2025"], "rolling_origin": summary["rolling_origin"], "spatial_generalization": summary["spatial_generalization"], "ablation": summary["ablation"]}


@app.get("/api/explainability")
def explainability() -> list[dict]:
    return public_json("importance.json")


@app.get("/api/calibration")
def calibration() -> dict:
    summary = public_json("summary.json")
    return {"curve": public_json("calibration.json"), "metrics": summary["occurrence_calibration"], "uncertainty": summary["bootstrap_mae_improvement_over_seasonal_naive"]}


@app.get("/api/data-quality")
def data_quality() -> dict:
    provenance = public_json("provenance.json")
    return {"provenance": provenance, "caveats": ["Police reports exclude or underrepresent some property-damage-only collisions.", "CHP handles freeway collisions.", "Police beats are administrative areas, not legal neighborhoods.", "Collision source does not contain verified point coordinates.", "Traffic counts are not used as exposure without a validated crosswalk."]}


@app.get("/api/forecast")
def forecast(limit: int = 2500) -> list[dict]:
    frame = pd.read_csv(require(ROOT / "experiments" / "historical" / "baseline-v0" / "weekly_predictions.csv"), parse_dates=["week"])
    return json.loads(frame.tail(max(1, min(limit, 10000))).to_json(orient="records", date_format="iso"))


@app.get("/api/importance")
def importance() -> list[dict]:
    return pd.read_csv(require(ROOT / "experiments" / "historical" / "baseline-v0" / "feature_importance.csv")).to_dict(orient="records")

# CRASH TEST

CRASH TEST is a reproducible research project for forecasting **relative,
next-week traffic-collision risk across San Diego Police beats**. It uses
official City of San Diego Police collision reports and publishes artifacts
only after a leakage-safe chronological evaluation.

## Verified current source snapshot

The local source snapshot contains **78,715 unique collision reports**, from
2015-03-19 through 2026-11-06, across **148 police beats**. This is a dynamic
official feed; reruns create a new provenance record with its exact checksum.

The present research task is aggregate beat-week prediction. It does not
predict individual crashes, infer GPS points, or treat selected traffic counts
as a citywide exposure measure without a validated crosswalk.

## Expanded verified results

On the held-out 2025 test period (7,696 beat-week rows), histogram gradient
boosting achieved **MAE 0.697**, versus **0.842** for Poisson regression and
**0.888** for the seasonal-naive baseline. Its top-decile risk ranking had
0.561 recall of the observed top decile and 3.587x lift. A calibrated
occurrence model achieved average precision **0.814** and Brier score **0.164**.
The bootstrap MAE improvement versus the seasonal baseline was 0.190
(95% CI 0.170 to 0.210). Exact artifacts are in `experiments/expanded/`.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,api]'
mkdir -p data/raw
# Download the official CSV from the source link below into data/raw/
PYTHONPATH=src python scripts/run_research.py
PYTHONPATH=src python scripts/run_expanded_research.py
PYTHONPATH=src uvicorn api.main:app --reload
```

In a second terminal:

```bash
cd dashboard
npm install
cp .env.example .env.local
npm run dev
```

## Data sources

- [San Diego Police traffic collisions - basic reports](https://data.sandiego.gov/datasets/traffic-collisions/): primary occurrence, beat, time, injury, fatality, and hit-and-run fields.
- [Traffic collision details](https://data.sandiego.gov/datasets/traffic-collision-details/): person-level companion source, cataloged but not currently merged.
- [Traffic ADT counts](https://data.sandiego.gov/datasets/traffic-adt-counts/): selected roadway counts, cataloged but not used without spatial reconciliation.
- [SWITRS / TIMS](https://tims.berkeley.edu/) and [Caltrans Traffic Census](https://dot.ca.gov/programs/traffic-operations/census): future-source candidates requiring access and compatible geography.

## Architecture

`data/raw` -> validation/deduplication -> complete police-beat weekly panel ->
lagged features -> rolling-origin, spatial-holdout, ablation, calibration, and
bootstrap evaluation -> saved JSON/CSV/PNG artifacts -> read-only FastAPI ->
Next.js dashboard.

Raw downloads, intermediate model frames, model artifacts, and secrets are
excluded from version control. The committed dashboard exports are aggregate
beat-week research outputs only; they contain no inferred crash coordinates.
See [methodology](docs/methodology.md) and [deployment](docs/deployment.md).

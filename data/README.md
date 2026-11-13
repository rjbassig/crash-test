# Data handling

Raw downloads and derived, row-level outputs are intentionally ignored by Git.
Run `python scripts/run_research.py` after downloading the official source files
into `data/raw/`.  The pipeline writes checksums, source metadata, validation
counts, model metrics, and figures to local artifact directories.

CrashCast's current research task is **next-week collision-count and relative
risk ranking by San Diego Police beat**. The official basic-report data does
not include coordinates, so this project does not infer point locations or
claim street-level prediction.

#!/usr/bin/env python3
"""Run the full reproducible CrashCast baseline experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from crashcast.research import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/pd_collisions_datasd.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run((root / args.raw).resolve() if not args.raw.is_absolute() else args.raw, root), indent=2))


if __name__ == "__main__":
    main()

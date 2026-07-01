#!/usr/bin/env python3
"""Bootstrap eval_set_imr.parquet from preview CSV or full IMR CSV if available."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
PREVIEW = ROOT / "data" / "processed" / "eval_set_imr_preview.csv"
FULL_CSV = ROOT / "data" / "raw" / "imr" / "independent-medical-review-imr-determinations-trend.csv"
OUT = ROOT / "data" / "processed" / "eval_set_imr.parquet"


def main() -> int:
    if FULL_CSV.exists():
        print(f"Building eval set from full IMR CSV ({FULL_CSV.name})...")
        return subprocess.call([sys.executable, str(ROOT / "scripts" / "02_stratify_eval_set.py")])

    if not PREVIEW.exists():
        print("ERROR: Neither full IMR CSV nor preview found.")
        print(f"  Expected: {FULL_CSV} or {PREVIEW}")
        return 1

    df = pd.read_csv(PREVIEW)
    if "was_overturned" not in df.columns and "Determination" in df.columns:
        df["was_overturned"] = df["Determination"].astype(str).str.contains(
            "Overturned", case=False, na=False
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"✓ Bootstrapped {len(df)} cases to {OUT} from preview CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

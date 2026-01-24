#!/usr/bin/env python3
"""
Rebuild training_index.csv from params.json files.

Assumes directory structure:
mumax_dataset_ku_by_block_disorder_phy_corrected_2/
├── run_00001/
│   └── params.json
├── run_00002/
│   └── params.json
...
├── run_01459/
│   └── params.json
"""

import json
import pandas as pd
from pathlib import Path
import re
import sys

# ---------------- CONFIG ----------------
BASE_DIR = Path("mumax_dataset_ku_by_block_disorder_phy_corrected_2")
OUTPUT_CSV = BASE_DIR / "training_index.csv"

# Parameters to keep (explicit whitelist)
KEEP_KEYS = [
    "run_id",
    "Dind",
    "sigma_nominal",
    "sigma_eff",
    "clamp_fraction",
    "sigma",
    "Temp",
    "Aex",
    "Ku_mean",
    "alpha",
    "Msat",
]

RUN_PATTERN = re.compile(r"run_(\d+)")
# ---------------------------------------


def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    rows = []
    seen_ids = set()
    run_dirs = sorted(d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("run_"))

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if not match:
            raise ValueError(f"Invalid run directory name: {run_dir.name}")

        folder_run_id = int(match.group(1))
        params_path = run_dir / "params.json"

        if not params_path.exists():
            raise FileNotFoundError(f"Missing params.json in {run_dir}")

        with open(params_path, "r") as f:
            params = json.load(f)

        if "run_id" not in params:
            raise KeyError(f"'run_id' missing in {params_path}")

        json_run_id = int(params["run_id"])

        # HARD consistency check
        if folder_run_id != json_run_id:
            raise ValueError(
                f"Run ID mismatch in {run_dir} | "
                f"folder: {folder_run_id} vs json: {json_run_id}"
            )

        if json_run_id in seen_ids:
            raise ValueError(f"Duplicate run_id detected: {json_run_id}")

        seen_ids.add(json_run_id)

        row = {key: params[key] for key in KEEP_KEYS if key in params}
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid runs were processed")

    df = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)

    print("====================================")
    print("training_index.csv successfully rebuilt")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total runs indexed: {len(df)}")
    print("====================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL ERROR:", e)
        sys.exit(1)

#!/usr/bin/env python3
"""
dataset_generator_methodB.py
--------------------------------------------
Generates randomized MuMax3 simulations with per-cell Ku1 disorder.

Each run folder contains:
 ├─ run_00001.mx3      (auto-generated MuMax3 script)
 ├─ final.ovf          (MuMax output magnetization)
 ├─ params.json        (metadata for training)
 ├─ mumax_stdout.txt   (stdout from MuMax3)
 ├─ mumax_stderr.txt   (stderr from MuMax3)
and the master index  dataset_index.csv  at the root.
"""

import subprocess
import pandas as pd
import numpy as np
import time
import json
import argparse
from pathlib import Path

# ---------------------- directories ----------------------
OUTPUT_BASE = Path("./mumax_dataset_ku_by_block_disorder")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
LOG_CSV = OUTPUT_BASE / "dataset_index.csv"
TRAINING_CSV = OUTPUT_BASE / "training_index.csv"

# ---------------------- simulation constants ----------------------
gridsize = (512, 512, 1)
cellsize = (4e-9, 4e-9, 1e-9)
n_seeds = 3
rng = np.random.default_rng(42)

# Parameter sweep ranges (replication study)
D_values = np.linspace(0.33e-3, 2.64e-3, 14)          # J/m^2
sigma_vals = np.linspace(0.00, 0.20, 9)          # dimensionless Ku dispersion
Temps = [0.0, 200.0,250.0, 300.0]                            # K

# Material parameter variability (± around mean)
Aex_values   = np.linspace(0.9e-11, 1.1e-11, 3)     # ±10%
Ku_mean_vals = np.linspace(2.1e5, 2.9e5, 3)         # ±15%
Msat_values  = np.linspace(1.08e6, 1.32e6, 3)       # ±10%
alpha_values = np.linspace(0.8, 1.2, 3)             # ±20%
blocksize = 8                                       # used for visualizing Ku grain scale

# ---------------------- helper functions ----------------------

mu0 = 4 * np.pi * 1e-7  # T m/A

import math

μ0 = 4 * math.pi * 1e-7  # vacuum permeability

def compute_micromagnetic_params(A, Ku, Ms, D):
    """Compute micromagnetic characteristic parameters."""
    Q = 2 * Ku / (μ0 * Ms**2)
    K_eff = Ku - 0.5 * μ0 * Ms**2
    K_eff_pos = max(K_eff, 1e3)  # avoid sqrt(0)
    lex = math.sqrt(2 * A / (μ0 * Ms**2))
    delta_dw = math.pi * math.sqrt(A / K_eff_pos)
    D_c = (4 / math.pi) * math.sqrt(A * K_eff_pos)
    D_ratio = D / D_c
    return Q, K_eff, lex, delta_dw, D_c, D_ratio

def print_run_diagnostics(A, Ku, Ms, D, cellX):
    Q, K_eff, lex, delta_dw, D_c, D_ratio = compute_micromagnetic_params(A, Ku, Ms, D)
    warn = []
    if cellX > lex:
        warn.append("⚠️ cell > ℓ_ex (undersampled exchange)")
    if delta_dw / cellX < 6:
        warn.append("⚠️ δ < 6 cells (DW under-resolved)")
    if abs(Q - 1) > 0.5:
        warn.append("⚠️ Q far from 1 (likely uniform or tilted state)")

    print(f"   → Q={Q:.2f}, Keff={K_eff:.2e}, ℓex={lex*1e9:.2f} nm, "
          f"δ={delta_dw*1e9:.2f} nm, D/Dc={D_ratio:.2f}")
    if warn:
        for w in warn:
            print("     " + w)




def is_valid_combo(Ku, Msat, Aex, Dind, sigma):
    """Return True if the combo is likely to form maze-like domains."""
    K_eff = Ku - 0.5 * mu0 * Msat**2

    # relaxed thresholds
    cond_anis  = abs(K_eff) < 0.8 * 0.5 * mu0 * Msat**2
    cond_DMI   = 0.1e-3 <= Dind <= 1.5e-3
    cond_sigma = 0.02 <= sigma <= 0.18

    return cond_anis and cond_DMI and cond_sigma

def write_mx3_with_regions(path, gridX=512, gridY=512,
                           cellX=4e-9, cellY=4e-9, cellZ=0.9e-9,
                           Msat=1.2e6, Aex=1.0e-11, alpha=1.0,
                           Dind=3.5e-3, meanKu=2.5e5, sigma=0.15,
                           blocksX=15, blocksY=15,
                           Temp=0.0,idx=0):
    """Write a .mx3 file implementing blockwise Ku disorder (≤256 regions)."""
    use_relax = Temp > 0.0
    lines = []

    # --- geometry & base params ---
    lines += [
        "// Auto-generated .mx3 (region-based Ku disorder)",
        f"gridX := {gridX}",
        f"gridY := {gridY}",
        f"cellX := {cellX}",
        f"cellY := {cellY}",
        "SetGridsize(gridX, gridY, 1)",
        f"SetCellsize({cellX}, {cellY}, {cellZ})",
        "SetPBC(1,1,0)",
        "",
        f"Msat  = {Msat}",
        f"Aex   = {Aex}",
        f"alpha = {alpha}",
        f"Dind  = {Dind}",
        "",
        "// --- Anisotropy with random block dispersion ---",
        f"meanKu := {meanKu}",
        f"sigma  := {sigma}",
        f"blocksX := {blocksX}",
        f"blocksY := {blocksY}",
        "baseBx := gridX / blocksX",
        "baseBy := gridY / blocksY",
        "blockWidth  := gridX * cellX / blocksX",
        "blockHeight := gridY * cellY / blocksY",
        "regionID := 1",
        "for bx := 0; bx < blocksX; bx++ {",
        "    for by := 0; by < blocksY; by++ {",
        "        Ku_local := meanKu * (1 + sigma * randnorm())",
        "        defregion(regionID,",
        "                  rect(blockWidth, blockHeight).Transl(",
        "                      (bx)*blockWidth - (blockWidth*(blocksX))/2 + blockWidth/2,",
        "                      (by)*blockHeight - (blockHeight*(blocksY))/2 + blockHeight/2,",
        "                      0))",
        "        Ku1.SetRegion(regionID, Ku_local)",
        "        regionID++",
        "    }",
        "}",
        "",
        "anisU = vector(0,0,1)",
        "B_ext = vector(0,0,0)",
        f"Temp  = {Temp}",
        "",
        "// --- Initial state & equilibrium solve ---",
        "m = RandomMag()",
        "Relax()" if use_relax else "Minimize()",
        "",
        "// --- Save outputs ---",
        f'SaveAs(m, "final_{idx}")',
        "tablesave()",
        ""
    ]
    path.write_text("\n".join(lines))

# ---------------------- start_gen loop ----------------------
def start_gen(max_runtime_minutes=120):
    start_time = time.time()
    max_runtime_s = max_runtime_minutes * 60

    # Resume from existing index if present
    if LOG_CSV.exists():
        existing = pd.read_csv(LOG_CSV)
        rows = existing.to_dict(orient="records")
        last_run = int(existing["run_id"].max()) if len(existing) > 0 else 0

        # build set of completed parameter tuples (D_mJpm2, sigma, Temp, seed)       This is to prevent re-running same parameters
        existing_keys = {
            (round(row["D_mJpm2"], 7), round(row["sigma"], 7), round(row["Temp"], 3))
            for row in existing.to_dict(orient="records")
        }
        print(f"Resuming from run_id={last_run}, {len(existing_keys)} parameter sets found.")
    else:
        last_run, rows, existing_keys = 0, [], set()
        print("Starting new dataset from scratch.")


    # For the new training CSV
    if TRAINING_CSV.exists():
        training_df = pd.read_csv(TRAINING_CSV)
        training_rows = training_df.to_dict(orient="records")
    else:
        training_rows = []

    run_id = last_run

    for D in D_values:
        for sigma in sigma_vals:

            for Ku_mean in Ku_mean_vals:
                for Msat in Msat_values:
                    for Aex in Aex_values:
                        for alpha in alpha_values:
                            # --------- check physical validity ----------
                            if not is_valid_combo(Ku_mean, Msat, Aex, D, sigma):
                                print(f"Skipping run: invalid physical combo D={D*1e3:.2f} σ={sigma:.2f}")

                                print_run_diagnostics(Aex, Ku_mean, Msat, D, cellsize[0])               

                                continue

                            for T in Temps:
                                for s in range(n_seeds):
                                    # --- 1. check time limit ---
                                    elapsed = time.time() - start_time
                                    if elapsed > max_runtime_s:
                                        print(f"Batch time limit reached ({elapsed/60:.1f} min). Saving progress...")
                                        pd.DataFrame(rows).to_csv(LOG_CSV, index=False)
                                        return


                                    run_id += 1
                                    run_dir = OUTPUT_BASE / f"run_{run_id:05d}"
                                    if run_dir.exists():
                                        continue  # already generated
                                    
                                    run_dir.mkdir(parents=True, exist_ok=True)


                                    # --- 2. write MuMax3 script referencing that OVF ---
                                    mx3_path = run_dir / f"run_{run_id:05d}.mx3"
                                    write_mx3_with_regions(
                                    mx3_path,
                                    gridX=512, gridY=512,
                                    Msat=Msat, Aex=Aex, alpha=alpha,
                                    Dind=D, meanKu=Ku_mean, sigma=sigma,
                                    blocksX=15, blocksY=15,
                                    Temp=T, idx=run_id
                                    )


                                    # --- 3. save parameters for record ---
                                    meta = dict(run_id=run_id, Dind=D, sigma=sigma, Temp=T,
                                                Aex=Aex, Ku_mean=Ku_mean, alpha=alpha,
                                                Msat=Msat, gridsize=gridsize, cellsize=cellsize)
                                    (run_dir / "params.json").write_text(json.dumps(meta, indent=2))

                                    # --- 4. run MuMax3 ---
                                    cmd = ["mumax3", "-o", str(run_dir), str(mx3_path)]
                                    t0 = time.time()
                                    proc = subprocess.run(cmd, capture_output=True, text=True)
                                    t1 = time.time()

                                    (run_dir/"mumax_stdout.txt").write_text(proc.stdout)
                                    (run_dir/"mumax_stderr.txt").write_text(proc.stderr)

                                    # --- 5. log summary row ---
                                    rows.append({
                                        "run_id": run_id,
                                        "D_mJpm2": D*1e3,
                                        "sigma": sigma,
                                        "Temp": T,
                                        "returncode": proc.returncode,
                                        "duration_s": t1 - t0,
                                        "run_dir": str(run_dir)
                                    })
                                    pd.DataFrame(rows).to_csv(LOG_CSV, index=False)

                                     # --- New Training Index Row ---
                                    training_rows.append({
                                        "run_id": run_id,
                                        "sample_idx": run_id,
                                        "realization": s,
                                        "Dind": D,
                                        "Ku1": Ku_mean,
                                        "Aex": Aex,
                                        "alpha": alpha,
                                        "Temp": T,
                                        "sigma": sigma,
                                        "mx3_path": str(mx3_path),
                                        "out_dir": str(run_dir),
                                        "returncode": proc.returncode,
                                        "timestamp_start": t0,
                                        "timestamp_end": t1
                                    })
                                    pd.DataFrame(training_rows).to_csv(TRAINING_CSV, index=False)

                                    print(f"[{run_id:05d}] D={D*1e3:.2f} σ={sigma:.2f} "
                                          f"T={T:.0f} rc={proc.returncode} ({t1-t0:.1f}s)")

    # --- 6. save master index ---
    pd.DataFrame(rows).to_csv(LOG_CSV, index=False)
    print(f"All {run_id} runs finished. Index written to {LOG_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timed batch MuMax3 dataset generator")
    parser.add_argument("--time", type=float, default=120, help="Max runtime in minutes (default 120)")
    args = parser.parse_args()
    start_gen(max_runtime_minutes=args.time)


# Run for 2 hours
# python3 k_blocks_dataset_gen.py --time 120
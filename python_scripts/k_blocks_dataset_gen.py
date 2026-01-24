#!/usr/bin/env python3
"""
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
from pathlib import Path
import math
import argparse

# ---------------------- directories ----------------------
OUTPUT_BASE = Path("./mumax_dataset_ku_by_block_disorder_phy_outside_clamp")
#OUTPUT_BASE = Path("./mumax_dataset_ku_by_block_disorder_phy_corrected_3")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
LOG_CSV = OUTPUT_BASE / "dataset_index.csv"
TRAINING_CSV = OUTPUT_BASE / "training_index.csv"
# ---------------------- thermal management ----------------------
WORK_BLOCK_SECONDS = 480 * 60    # 40 minutos de trabajo continuo
SLEEP_SECONDS      = 20 * 60     # 4 minutos de descanso


# ---------------------- simulation constants ----------------------
gridsize = (512, 512, 1)
cellsize = (4e-9, 4e-9, 0.9e-9)
n_seeds = 1
#rng = np.random.default_rng(42)
# --- Sampling Ranges ---
# The CNN will learn to predict values anywhere inside these ranges
# --- Sampling Ranges (Continuous Uniform) ---
range_D     = (1.6e-3, 2.5e-3)
range_sigma = (0.17, 0.25)
Temps = [0.0]

# --- Base Material Parameters (Will have Jitter added) ---
BASE_Aex   = 3.1e-11
BASE_Ku    = 1.6e6
BASE_Msat  = 1.445e6
BASE_alpha = 1.0
blocksize = 16                                       # used for visualizing Ku grain scale

# ---------------------- helper functions ----------------------

def atomic_to_csv(df, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


mu0 = 4 * np.pi * 1e-7  # T m/A

def generate_Ku_blocks(meanKu, sigma_nominal, Msat,
                       blocksX, blocksY):
    """
    Generate block-wise Ku with Gaussian disorder + physical clamp.
    Returns:
        Ku_blocks (np.ndarray shape [n_blocks])
        sigma_eff (float)
        clamp_fraction (float)
    """
    n_blocks = blocksX * blocksY

    # Gaussian disorder
    z = np.random.normal(0.0, 1.0, size=n_blocks)
    Ku_blocks = meanKu * (1.0 + sigma_nominal * z)

    # Physical clamp
    Kd_limit = 0.5 * mu0 * Msat**2
    safe_min_Ku = 1.05 * Kd_limit

    clamped = Ku_blocks < safe_min_Ku
    Ku_blocks[clamped] = safe_min_Ku

    # Effective sigma (relative)
    sigma_eff = np.std(Ku_blocks) / np.mean(Ku_blocks)
    clamp_fraction = clamped.mean()

    return Ku_blocks, sigma_eff, clamp_fraction


def compute_micromagnetic_params(A, Ku, Ms, D):
    """Compute micromagnetic characteristic parameters."""
    Q = 2 * Ku / (mu0 * Ms**2)
    K_eff = Ku - 0.5 * mu0 * Ms**2
    K_eff_pos = max(K_eff, 1e3)  # avoid sqrt(0)
    lex = math.sqrt(2 * A / (mu0 * Ms**2))
    delta_dw = math.pi * math.sqrt(A / K_eff_pos)
    D_c = (4 / math.pi) * math.sqrt(A * K_eff_pos)
    D_ratio = D / D_c
    return Q, K_eff, lex, delta_dw, D_c, D_ratio

def print_run_diagnostics(A, Ku, Ms, D, cellX):
    Q, K_eff, lex, delta_dw, D_c, D_ratio = compute_micromagnetic_params(A, Ku, Ms, D)
    warn = []
    if cellX > lex:
        warn.append("cell > ℓ_ex (undersampled exchange)")
    if delta_dw / cellX < 6:
        warn.append("δ < 6 cells (DW under-resolved)")
    if abs(Q - 1) > 0.5:
        warn.append("Q far from 1 (likely uniform or tilted state)")

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
                           Dind=3.5e-3, meanKu=2.5e5,
                           Ku_blocks=None,
                           blocksX=16, blocksY=16,
                           Temp=0.0, idx=0):
    """
    Write a .mx3 file implementing blockwise Ku disorder
    using Ku values generated externally in Python.
    """
    assert Ku_blocks is not None, "Ku_blocks must be provided"
    assert len(Ku_blocks) == blocksX * blocksY, \
        "Ku_blocks length must equal blocksX * blocksY"

    use_relax = True
    lines = []

    # --- geometry & base params ---
    lines += [
        "// Auto-generated .mx3 (region-based Ku disorder, Python-controlled)",
        f"gridX := {gridX}",
        f"gridY := {gridY}",
        f"cellX := {cellX}",
        f"cellY := {cellY}",
        "SetGridsize(gridX, gridY, 1)",
        f"SetCellsize({cellX}, {cellY}, {cellZ})",
        "SetPBC(1,1,0)",
        "",
        f"M_val := {Msat}",
        "Msat = M_val",
        f"Aex   = {Aex}",
        f"alpha = {alpha}",
        f"Dind  = {Dind}",
        "",
        "// --- Anisotropy with externally generated block dispersion ---",
        f"blocksX := {blocksX}",
        f"blocksY := {blocksY}",
        "blockWidth  := gridX * cellX / blocksX",
        "blockHeight := gridY * cellY / blocksY",
        "regionID := 0",
    ]

    # --- define regions and assign Ku explicitly ---
    ku_idx = 0
    for bx in range(blocksX):
        for by in range(blocksY):
            Ku_val = Ku_blocks[ku_idx]
            lines += [
                "defregion(regionID,",
                "  rect(blockWidth*1.01, blockHeight*1.01).Transl(",
                f"    ({bx})*blockWidth - (blockWidth*(blocksX))/2 + blockWidth/2,",
                f"    ({by})*blockHeight - (blockHeight*(blocksY))/2 + blockHeight/2,",
                "    0))",
                f"Ku1.SetRegion(regionID, {Ku_val})",
                "regionID++",
            ]
            ku_idx += 1

    lines += [
        "",
        "anisU = vector(0,0,1)",
        "B_ext = vector(0,0,0)",
        f"Temp  = {Temp}",
        "",
        "// --- Initial state & equilibrium solve ---",
        "m = RandomMag()",
        "Run(80e-9) // initial relaxation",
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
    
    last_sleep_time = start_time

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
    
    while True:
        # Uniform Random Sampling for Target Variables
            D     = np.random.uniform(range_D[0], range_D[1])
            sigma = np.random.uniform(range_sigma[0], range_sigma[1])

            # Gaussian Jitter for Material Constants (±2% std dev)
            # This makes the CNN robust against small material variations
            #Aex   = np.random.normal(BASE_Aex,  BASE_Aex * 0.02)
            #Ku_mean    = np.random.normal(BASE_Ku,   BASE_Ku * 0.02)
            #Msat  = np.random.normal(BASE_Msat, BASE_Msat * 0.02)
            #alpha = BASE_alpha # Alpha usually fixed, or jitter 1% if desired

            Aex   = BASE_Aex 
            Ku_mean    = BASE_Ku 
            Msat  = BASE_Msat 
            alpha = BASE_alpha # Alpha usually fixed, or jitter 1% if desired

            # --------- check physical validity ----------
            #if not is_valid_combo(Ku_mean, Msat, Aex, D, sigma):
            #    print(f"Skipping run: invalid physical combo D={D*1e3:.2f} σ={sigma:.2f}")
#
            #    print_run_diagnostics(Aex, Ku_mean, Msat, D, cellsize[0])               
#
            #    continue

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
                    
                    Ku_blocks, sigma_eff, clamp_fraction = generate_Ku_blocks(
                        meanKu=Ku_mean,
                        sigma_nominal=sigma,
                        Msat=Msat,
                        blocksX=16,
                        blocksY=16
                    )
                    
                    write_mx3_with_regions(
                        mx3_path,
                        gridX=512, gridY=512,
                        Msat=Msat, Aex=Aex, alpha=alpha,
                        Dind=D, meanKu=Ku_mean,
                        Ku_blocks=Ku_blocks,
                        blocksX=16, blocksY=16,
                        Temp=T, idx=run_id
                    )



                    # --- 3. save parameters for record ---
                    meta = dict(
                        run_id=run_id,
                        Dind=D,
                        sigma_nominal=sigma,
                        sigma_eff=sigma_eff,
                        clamp_fraction=clamp_fraction,
                        Temp=T,
                        Aex=Aex,
                        Ku_mean=Ku_mean,
                        alpha=alpha,
                        Msat=Msat,
                        gridsize=gridsize,
                        cellsize=cellsize
                    )

                    
                    (run_dir / "params.json").write_text(json.dumps(meta, indent=2))
                    
                    # --- thermal throttling: periodic sleep ---
                    now = time.time()
                    time_since_sleep = now - last_sleep_time
                    
                    if time_since_sleep > WORK_BLOCK_SECONDS:
                        remaining = max_runtime_s - (now - start_time)
                        if remaining <= 0:
                            print("Time limit reached during thermal pause window.")
                            return
                    
                        sleep_time = min(SLEEP_SECONDS, remaining)
                        print(f"Thermal pause: sleeping for {sleep_time/60:.1f} min...")
                        time.sleep(sleep_time)
                        last_sleep_time = time.time()

                    

                    # --- 4. run MuMax3 ---
                    cmd = ["mumax3", "-o", str(run_dir), str(mx3_path)]
                    t0 = time.time()
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    t1 = time.time()

                    (run_dir/"mumax_stdout.txt").write_text(proc.stdout)
                    (run_dir/"mumax_stderr.txt").write_text(proc.stderr)
                    
                    final_ovf = run_dir / f"final_{run_id}.ovf"   # because SaveAs(m, "final_{idx}")
                    if proc.returncode != 0 or not final_ovf.exists():
                        print(f"[FAIL] run {run_id:05d} rc={proc.returncode} final_exists={final_ovf.exists()}")
                        # Option 1: stop the whole batch (recommended)
                        raise RuntimeError("MuMax failed; stopping to prevent corrupt/incomplete dataset.")
                        # Option 2: continue but DO NOT add to training_rows


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
                    
                    #pd.DataFrame(rows).to_csv(LOG_CSV, index=False)

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
                        "sigma_nominal": sigma,
                        "sigma_eff": sigma_eff,
                        "clamp_fraction": clamp_fraction,
                        "mx3_path": str(mx3_path),
                        "out_dir": str(run_dir),
                        "returncode": proc.returncode,
                        "timestamp_start": t0,
                        "timestamp_end": t1
                    })
                    #pd.DataFrame(training_rows).to_csv(TRAINING_CSV, index=False)
                    atomic_to_csv(pd.DataFrame(rows), LOG_CSV)
                    atomic_to_csv(pd.DataFrame(training_rows), TRAINING_CSV)


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
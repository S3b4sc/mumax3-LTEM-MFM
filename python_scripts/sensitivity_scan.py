#!/usr/bin/env python3
"""
sensitivity_scan_mumax.py

Automates a sensitivity scan:
 - samples parameter space (LHS or grid)
 - for each sample creates `n_realizations` different Ku maps + mx3
 - launches mumax3, waits for completion, captures stdout/stderr
 - logs run metadata to a CSV
"""

import subprocess
import pandas as pd
import numpy as np
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Sequence
import json
import sys

# -------------------------
# OUTPUT / file locations
# -------------------------
OUTPUT_BASE = Path("./mumax_files/logs")
LOG_CSV = OUTPUT_BASE / "sensitivity_simulation_log.csv"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# -------------------------
# Simulation control (constants)
# -------------------------
max_time = 5e-9            # physical simulation maximum time (seconds) - used only if run(), ignored for Minimize/Relax
autosave_interval = 1e-9   # (s) autosave magnetization frequency inside the mx3 script
tableautosave_interval = 1e-10  # (s) how often MuMax writes table.txt
gridsize = (256, 256, 1)   # default smaller patch for sensitivity pilot (faster). change to (512,512,1) later if desired
cellsize = (4e-9, 4e-9, 0.9e-9)
blocksize = 8              # block size for Ku map (8x8 blocks as in the paper)

# -------------------------
# Parameter bounds for sensitivity sampling
# (you can edit these ranges)
# Units: Dind in J/m^2 (e.g. 1e-3 = 1 mJ/m^2), Ku in J/m^3, Aex in J/m, alpha dimensionless, Temp in K, sigma relative
# -------------------------
param_bounds = {
    "Dind": (0.0, 1.5e-3),      # 0 - 1.5 mJ/m^2
    "Ku1":  (2.5e5, 8.0e5),     # 2.5e5 - 8e5 J/m^3
    "Aex":  (7e-12, 20e-12),    # 7 - 20 pJ/m
    "alpha":(0.1, 1.0),         # damping
    "Temp": (0.0, 300.0),       # temperature (K) - allow 0 or finite
    "sigma":(0.0, 0.2)          # relative Ku disorder (0 - 0.2)
}

# -------------------------
# Sampling & experiment design
# -------------------------
SAMPLING_METHOD = "lhs"   # "lhs" or "grid"
n_parameter_samples = 90  # number of distinct parameter points (LHS samples). start small (100-500) for pilot
n_realizations = 3         # number of random realizations (Ku map + randomMag) per sampled parameter set
seed = 42                  # reproducibility for LHS

# -------------------------
# Helper: small logger
# -------------------------
def log(msg: str, level: str = "INFO", run_id: Optional[int] = None) -> None:
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{t}] {level}"
    if run_id is not None:
        prefix += f" (run {run_id})"
    print(f"{prefix}: {msg}")

# -------------------------
# Sampling functions
# -------------------------
def lhs_sample(bounds: Dict[str, tuple], n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simple Latin Hypercube Sampling returning a DataFrame with n rows and columns = bounds keys."""
    keys = list(bounds.keys())
    k = len(keys)
    seg = rng.random((n, k))
    # Make standard LHS: for each dimension, shuffle the n stratified positions
    samples = np.zeros((n, k))
    for j, key in enumerate(keys):
        lo, hi = bounds[key]
        perm = rng.permutation(n)
        stratified = (perm + seg[:, j]) / n
        samples[:, j] = lo + stratified * (hi - lo)
    df = pd.DataFrame(samples, columns=keys)
    return df

def grid_sample(bounds: Dict[str, tuple], grid_pts_per_dim: int = 5) -> pd.DataFrame:
    """Cartesian grid sampling (may explode combinatorially)."""
    import itertools
    keys = list(bounds.keys())
    lists = []
    for k in keys:
        lo, hi = bounds[k]
        lists.append(np.linspace(lo, hi, grid_pts_per_dim))
    combos = list(itertools.product(*lists))
    df = pd.DataFrame(combos, columns=keys)
    return df

# -------------------------
# Ku-map generator (blocky Gaussian)
# -------------------------
def generate_ku_map(ku_mean: float, sigma_rel: float, gridsize: Sequence[int], blocksize: int, rng: np.random.Generator) -> np.ndarray:
    nx, ny = gridsize[0], gridsize[1]
    bx = nx // blocksize
    by = ny // blocksize
    # small safeguard: if gridsize not divisible by blocksize, use ceil and then crop
    if nx % blocksize != 0 or ny % blocksize != 0:
        bx = int(np.ceil(nx / blocksize))
        by = int(np.ceil(ny / blocksize))
    block_vals = rng.normal(loc=ku_mean, scale=sigma_rel * ku_mean, size=(bx, by))
    ku_map = np.kron(block_vals, np.ones((blocksize, blocksize)))
    # crop to exact nx,ny
    ku_map = ku_map[:nx, :ny]
    return ku_map

# -------------------------
# MX3 writer
# -------------------------
def write_mx3_file(path: Path, params: Dict[str, Any], ku_map_filename: Optional[str],
                   gridsize=(256,256,1), cellsize=(4e-9,4e-9,0.9e-9),
                   autosave_interval: float = autosave_interval,
                   tableautosave_interval: float = tableautosave_interval,
                   max_time: float = max_time) -> None:
    nx, ny, nz = gridsize
    dx, dy, dz = cellsize
    B = params.get("B_ext", [0.0,0.0,0.0])
    Dind = params.get("Dind", 0.0)
    Ku1 = params.get("Ku1", 3e5)
    Aex = params.get("Aex", 10e-12)
    alpha = params.get("alpha", 0.8)
    Temp = params.get("Temp", 0.0)

    # Decide relax mode: Temp>0 -> Relax (with thermal noise). Temp==0 -> Minimize() (faster)
    use_relax = bool(Temp > 0.0)

    # muMax code: loads Ku map file if provided (the exact load command may depend on muMax version)
    # we use a simple approach: load a text table into a scalar field. If your MuMax version needs different commands, adapt here.
    if ku_map_filename is not None:
        ku_load_lines = f"""
// load Ku map generated externally (blocky values)
//Ku_tab := LoadTable("{ku_map_filename}")
//Ku1 = ScalarFieldFromTable(Ku_tab)    // NOTE: check this command for your MuMax3 version
Ku1 = {Ku1}
"""
    else:
        ku_load_lines = f"Ku1 = {Ku1}\n"

    mx3 = f"""// Auto-generated by sensitivity_scan_mumax.py
SetGridsize({nx},{ny},{nz})
SetCellsize({dx},{dy},{dz})
Msat = {params.get('Msat', 1.446e6)}
Aex = {Aex}
alpha = {alpha}
Dind = {Dind}
anisU = vector(0,0,1)
B_ext = vector({B[0]},{B[1]},{B[2]})
Temp = {Temp}

{ku_load_lines}
// Table & autosave settings
TableAdd(E_total)
TableAdd(MaxTorque)
tableautosave({tableautosave_interval})
autosave(m, {autosave_interval})

// Random init and run
m = RandomMag()
"""

    if use_relax:
        mx3 += "Relax()\n"
    else:
        # use energy minimization for T=0
        mx3 += "Minimize()\n"

    mx3 += 'SaveAs(m, "final.ovf")\n'
    mx3 += '\n// End of generated script\n'
    path.write_text(mx3)

# -------------------------
# Main driver
# -------------------------
def start_sensitivity_scan():
    rng = np.random.default_rng(seed)

    # Build parameter samples
    if SAMPLING_METHOD == "lhs":
        samples_df = lhs_sample(param_bounds, n_parameter_samples, rng)
    else:
        samples_df = grid_sample(param_bounds, grid_pts_per_dim=5)

    log(f"Prepared {len(samples_df)} parameter samples (method={SAMPLING_METHOD})", level="INFO")

    # Prepare logging CSV
    if not LOG_CSV.exists():
        df0 = pd.DataFrame(columns=[
            "run_id", "sample_idx", "realization", "Dind", "Ku1", "Aex", "alpha", "Temp", "sigma",
            "mx3_path", "ku_map_path", "out_dir", "returncode", "stdout_file", "stderr_file", "timestamp_start", "timestamp_end"
        ])
        df0.to_csv(LOG_CSV, index=False)

    run_id = 0
    for sample_idx, row in samples_df.iterrows():
        # read sample parameters
        Dind = float(row["Dind"])
        Ku1 = float(row["Ku1"])
        Aex = float(row["Aex"])
        alpha = float(row["alpha"])
        Temp = float(row["Temp"])
        sigma_rel = float(row["sigma"])

        for realization in range(n_realizations):
            run_id += 1
            run_dir = OUTPUT_BASE / f"run_{run_id:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # generate Ku map for this realization and save as txt and npy
            ku_map = generate_ku_map(Ku1, sigma_rel, gridsize, blocksize, rng)
            ku_map_txt = run_dir / "Ku_map.txt"
            ku_map_npy = run_dir / "Ku_map.npy"
            np.savetxt(ku_map_txt, ku_map, fmt="%.6e")
            np.save(ku_map_npy, ku_map)

            # build params dict for mx3 writer
            params = {
                "Dind": Dind,
                "Ku1": Ku1,    # mean (used if ku map not loaded)
                "Aex": Aex,
                "alpha": alpha,
                "B_ext": [0.0, 0.0, 0.0],  # keep B = 0 for sensitivity pilot
                "Msat": 1.446e6,
                "Temp": Temp
            }

            mx3_path = run_dir / f"run_{run_id:05d}.mx3"
            # write mx3 that references ku_map relative path
            write_mx3_file(mx3_path, params, ku_map_filename=str(ku_map_txt.name),
                           gridsize=gridsize, cellsize=cellsize,
                           autosave_interval=autosave_interval,
                           tableautosave_interval=tableautosave_interval,
                           max_time=max_time)

            # copy a small README or param json to run dir
            meta = {
                "Dind": Dind, "Ku1": Ku1, "Aex": Aex, "alpha": alpha, "Temp": Temp, "sigma": sigma_rel,
                "gridsize": gridsize, "cellsize": cellsize, "blocksize": blocksize
            }
            (run_dir / "params.json").write_text(json.dumps(meta, indent=2))

            # Launch mumax3 and wait for completion (synchronous)
            cmd = ["mumax3", "-o", str(run_dir), str(mx3_path)]
            log(f"Launching MuMax3: run_id={run_id} sample={sample_idx} realization={realization}", run_id=run_id)
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                returncode = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
            except FileNotFoundError:
                log("Could not find 'mumax3' executable. Make sure it's on PATH.", level="ERROR", run_id=run_id)
                return
            t1 = time.time()

            # Save stdout/stderr to files
            stdout_file = run_dir / "mumax_stdout.txt"
            stderr_file = run_dir / "mumax_stderr.txt"
            stdout_file.write_text(stdout)
            stderr_file.write_text(stderr)

            # Append a row to CSV log
            row_out = {
                "run_id": run_id,
                "sample_idx": int(sample_idx),
                "realization": int(realization),
                "Dind": Dind, "Ku1": Ku1, "Aex": Aex, "alpha": alpha, "Temp": Temp, "sigma": sigma_rel,
                "mx3_path": str(mx3_path),
                "ku_map_path": str(ku_map_txt),
                "out_dir": str(run_dir),
                "returncode": int(returncode),
                "stdout_file": str(stdout_file),
                "stderr_file": str(stderr_file),
                "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
            }
            df_log = pd.DataFrame([row_out])
            df_log.to_csv(LOG_CSV, mode="a", header=False, index=False)

            log(f"Run {run_id} finished (rc={returncode}) in {t1 - t0:.1f}s", run_id=run_id)

    log("All samples processed.", level="INFO")

if __name__ == "__main__":
    start_sensitivity_scan()

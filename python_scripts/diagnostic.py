#!/usr/bin/env python3
"""
Run this script to diagnose your PINN training issues.
Make sure to update the paths to match your setup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ==============================================================================

CSV_PATH = Path("../mumax_dataset_ku_by_block_disorder_phy_corrected_3/training_index.csv")
IMG_DIR = Path("../images/corrected_3")

# Surrogate checkpoints
SURR_FFT_PATH = Path("./outputs_noVariation_2372imgs/FFT_corr_/checkpoints/surrogate_fft_FFT_corr_.pt")
SURR_CORR_PATH = Path("./outputs_noVariation_2372imgs/FFT_corr_/checkpoints/surrogate_corr_FFT_corr_.pt")

IMG_PREFIX = "mz_binary_"
IMG_EXT = ".png"
COL_D = "Dind"
COL_SIGMA = "sigma_nominal"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# HELPER FUNCTIONS (from your code)
# ==============================================================================

def load_16bit_image(path: Path) -> np.ndarray:
    """Load 16-bit grayscale PNG preserving full dynamic range."""
    img = Image.open(path)
    if img.mode == 'I;16':
        arr = np.array(img, dtype=np.uint16)
        arr = arr.astype(np.float32) / 65535.0
    elif img.mode == 'I':
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    elif img.mode == 'L':
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected image mode: {img.mode}")
    return arr


def compute_fft_observables(img_batch, n_radial_bins=64):
    """Your FFT observables function."""
    x = img_batch[:, 0, :, :]
    x = 2.0 * x - 1.0

    B, H, W = x.shape
    eps = 1e-8

    x = x - x.mean(dim=(1, 2), keepdim=True)

    F_t = torch.fft.fft2(x)
    F_t = torch.fft.fftshift(F_t, dim=(1, 2))
    P = torch.log1p(torch.abs(F_t) ** 2)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij"
    )
    k = torch.sqrt(xx**2 + yy**2)

    k_flat = k.reshape(-1)
    P_flat = P.reshape(B, -1)

    k_max = k_flat.max()
    bin_edges = torch.linspace(0.0, k_max, n_radial_bins + 1, device=x.device)
    k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    S_bins = []
    for i in range(n_radial_bins):
        mask = (k_flat >= bin_edges[i]) & (k_flat < bin_edges[i + 1])
        mask = mask.unsqueeze(0)
        denom = mask.sum(dim=1).clamp_min(1.0)
        S_i = (P_flat * mask).sum(dim=1) / denom
        S_bins.append(S_i)

    S_bins = torch.stack(S_bins, dim=1)
    S_norm = S_bins / (S_bins.sum(dim=1, keepdim=True) + eps)

    k_c = k_centers.unsqueeze(0)
    mu1 = (S_norm * k_c).sum(dim=1)
    mu2 = (S_norm * (k_c - mu1.unsqueeze(1))**2).sum(dim=1)
    mu3 = ((S_norm * (k_c - mu1.unsqueeze(1))**3).sum(dim=1) / (mu2.sqrt()**3 + eps))
    mu4 = ((S_norm * (k_c - mu1.unsqueeze(1))**4).sum(dim=1) / (mu2**2 + eps))

    peak_val, _ = S_bins.max(dim=1)
    background = S_bins.mean(dim=1)
    peak_ratio = peak_val / (background + eps)

    high_k_mask = k_c > 0.6 * k_max
    high_k_energy = (S_norm * high_k_mask).sum(dim=1)

    Phi_fft = torch.cat([
        S_norm,
        mu1.unsqueeze(1),
        mu2.unsqueeze(1),
        mu3.unsqueeze(1),
        mu4.unsqueeze(1),
        peak_ratio.unsqueeze(1),
        high_k_energy.unsqueeze(1)
    ], dim=1)

    return Phi_fft


def compute_corr_observables(img_batch, n_radial_bins=64):
    """Your correlation observables function."""
    x = img_batch[:, 0, :, :]
    x = 2.0 * x - 1.0

    B, H, W = x.shape
    eps = 1e-8

    x = x - x.mean(dim=(1, 2), keepdim=True)

    Fx = torch.fft.fft2(x)
    P = Fx.real**2 + Fx.imag**2
    C = torch.fft.ifft2(P).real

    C0 = C[:, 0, 0].clamp_min(eps)
    C = C / C0[:, None, None]

    yy = torch.fft.fftfreq(H, device=x.device) * H
    xx = torch.fft.fftfreq(W, device=x.device) * W
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)

    r_int = torch.round(r).long()
    max_r = min(n_radial_bins, r_int.max().item())

    C_flat = C.reshape(B, -1)
    r_flat = r_int.reshape(-1)

    Cr = []
    for rr in range(1, max_r + 1):
        mask = (r_flat == rr).float()
        denom = mask.sum().clamp_min(1.0)
        Cr_r = (C_flat * mask).sum(dim=1) / denom
        Cr.append(Cr_r)

    Cr = torch.stack(Cr, dim=1)

    if Cr.shape[1] < n_radial_bins:
        pad = torch.zeros(B, n_radial_bins - Cr.shape[1], device=x.device)
        Cr = torch.cat([Cr, pad], dim=1)

    sign = torch.sign(Cr)
    zero_cross = (sign[:, :-1] * sign[:, 1:] < 0)

    L_corr = torch.where(
        zero_cross.any(dim=1),
        zero_cross.float().argmax(dim=1) + 1,
        torch.full((B,), float(n_radial_bins), device=x.device)
    )

    Phi_corr = torch.cat([Cr, L_corr.unsqueeze(1)], dim=1)

    return Phi_corr


class SurrogatePhysics(nn.Module):
    def __init__(self, phi_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, phi_dim)
        )

    def forward(self, params):
        return self.net(params)


# ==============================================================================
# DIAGNOSTIC 1: Surrogate Quality Check
# ==============================================================================

def check_surrogate_quality():
    print("\n" + "="*60)
    print("DIAGNOSTIC 1: SURROGATE MODEL QUALITY")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df.set_index("run_id", inplace=True)
    
    # Get unique (D, sigma) pairs
    img_files = sorted(IMG_DIR.glob(f"{IMG_PREFIX}*{IMG_EXT}"))
    
    param_to_images = {}
    for img_path in img_files:
        try:
            run_id = int(img_path.stem.replace(IMG_PREFIX, ""))
        except ValueError:
            continue
        if run_id not in df.index:
            continue
        
        row = df.loc[run_id]
        D = float(row[COL_D])
        sigma = float(row[COL_SIGMA])
        key = (round(D, 8), round(sigma, 6))
        
        if key not in param_to_images:
            param_to_images[key] = []
        param_to_images[key].append(img_path)
    
    print(f"Found {len(param_to_images)} unique (D, sigma) combinations")
    
    # Load surrogates
    ckpt_fft = torch.load(SURR_FFT_PATH, map_location=device)
    ckpt_corr = torch.load(SURR_CORR_PATH, map_location=device)
    
    phi_fft_dim = ckpt_fft["phi_dim"]
    phi_corr_dim = ckpt_corr["phi_dim"]
    
    Phi_fft_mean = ckpt_fft["Phi_mean"].to(device)
    Phi_fft_std = ckpt_fft["Phi_std"].to(device)
    Phi_corr_mean = ckpt_corr["Phi_mean"].to(device)
    Phi_corr_std = ckpt_corr["Phi_std"].to(device)
    
    surrogate_fft = SurrogatePhysics(phi_dim=phi_fft_dim).to(device)
    surrogate_fft.load_state_dict(ckpt_fft["state_dict"])
    surrogate_fft.eval()
    
    surrogate_corr = SurrogatePhysics(phi_dim=phi_corr_dim).to(device)
    surrogate_corr.load_state_dict(ckpt_corr["state_dict"])
    surrogate_corr.eval()
    
    # Evaluate on all (D, sigma) pairs
    all_phi_fft_true = []
    all_phi_fft_pred = []
    all_phi_corr_true = []
    all_phi_corr_pred = []
    all_params = []
    
    print("\nComputing Phi for all parameter combinations...")
    
    with torch.no_grad():
        for (D, sigma), img_paths in list(param_to_images.items())[:200]:  # Limit for speed
            # Load images
            imgs = []
            for p in img_paths[:5]:  # Max 5 images per param
                arr = load_16bit_image(p)
                img = torch.from_numpy(arr).unsqueeze(0).float()
                img = img.repeat(3, 1, 1)
                imgs.append(img)
            
            imgs = torch.stack(imgs).to(device)
            
            # Compute true Phi (average over realizations)
            phi_fft_true = compute_fft_observables(imgs).mean(dim=0)
            phi_corr_true = compute_corr_observables(imgs).mean(dim=0)
            
            # Normalize
            phi_fft_true_norm = (phi_fft_true - Phi_fft_mean) / Phi_fft_std
            phi_corr_true_norm = (phi_corr_true - Phi_corr_mean) / Phi_corr_std
            
            # Predict from surrogate
            params = torch.tensor([D, sigma], dtype=torch.float32, device=device)
            phi_fft_pred = surrogate_fft(params.unsqueeze(0)).squeeze(0)
            phi_corr_pred = surrogate_corr(params.unsqueeze(0)).squeeze(0)
            
            all_phi_fft_true.append(phi_fft_true_norm.cpu())
            all_phi_fft_pred.append(phi_fft_pred.cpu())
            all_phi_corr_true.append(phi_corr_true_norm.cpu())
            all_phi_corr_pred.append(phi_corr_pred.cpu())
            all_params.append([D, sigma])
    
    all_phi_fft_true = torch.stack(all_phi_fft_true)
    all_phi_fft_pred = torch.stack(all_phi_fft_pred)
    all_phi_corr_true = torch.stack(all_phi_corr_true)
    all_phi_corr_pred = torch.stack(all_phi_corr_pred)
    all_params = np.array(all_params)
    
    # Compute metrics
    def compute_r2(pred, true):
        ss_res = ((pred - true) ** 2).sum()
        ss_tot = ((true - true.mean(dim=0)) ** 2).sum()
        return (1 - ss_res / ss_tot).item()
    
    r2_fft = compute_r2(all_phi_fft_pred, all_phi_fft_true)
    r2_corr = compute_r2(all_phi_corr_pred, all_phi_corr_true)
    
    mse_fft = F.mse_loss(all_phi_fft_pred, all_phi_fft_true).item()
    mse_corr = F.mse_loss(all_phi_corr_pred, all_phi_corr_true).item()
    
    print(f"\n--- FFT Surrogate ---")
    print(f"  MSE: {mse_fft:.6f}")
    print(f"  R²:  {r2_fft:.4f}")
    
    print(f"\n--- Corr Surrogate ---")
    print(f"  MSE: {mse_corr:.6f}")
    print(f"  R²:  {r2_corr:.4f}")
    
    # Warnings
    print("\n--- Assessment ---")
    if r2_fft < 0.5:
        print("⚠️  FFT surrogate R² < 0.5: Physics loss from FFT is likely HARMFUL")
    elif r2_fft < 0.8:
        print("⚠️  FFT surrogate R² < 0.8: Moderate quality, use with caution")
    else:
        print("✓  FFT surrogate quality acceptable")
    
    if r2_corr < 0.5:
        print("⚠️  Corr surrogate R² < 0.5: Physics loss from Corr is likely HARMFUL")
    elif r2_corr < 0.8:
        print("⚠️  Corr surrogate R² < 0.8: Moderate quality, use with caution")
    else:
        print("✓  Corr surrogate quality acceptable")
    
    return {
        "r2_fft": r2_fft,
        "r2_corr": r2_corr,
        "mse_fft": mse_fft,
        "mse_corr": mse_corr,
        "params": all_params,
        "phi_fft_true": all_phi_fft_true,
        "phi_fft_pred": all_phi_fft_pred,
        "phi_corr_true": all_phi_corr_true,
        "phi_corr_pred": all_phi_corr_pred
    }


# ==============================================================================
# DIAGNOSTIC 2: Identifiability Check
# ==============================================================================

def check_identifiability():
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: PARAMETER IDENTIFIABILITY")
    print("="*60)
    
    df = pd.read_csv(CSV_PATH)
    df.set_index("run_id", inplace=True)
    
    img_files = sorted(IMG_DIR.glob(f"{IMG_PREFIX}*{IMG_EXT}"))
    
    # Sample images
    np.random.seed(42)
    sample_files = np.random.choice(img_files, min(150, len(img_files)), replace=False)
    
    params_list = []
    phi_fft_list = []
    phi_corr_list = []
    
    print("Computing Phi for sampled images...")
    
    with torch.no_grad():
        for img_path in sample_files:
            try:
                run_id = int(img_path.stem.replace(IMG_PREFIX, ""))
            except ValueError:
                continue
            if run_id not in df.index:
                continue
            
            row = df.loc[run_id]
            D = float(row[COL_D])
            sigma = float(row[COL_SIGMA])
            
            arr = load_16bit_image(img_path)
            img = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            img = img.repeat(1, 3, 1, 1).to(device)
            
            phi_fft = compute_fft_observables(img).squeeze(0).cpu().numpy()
            phi_corr = compute_corr_observables(img).squeeze(0).cpu().numpy()
            
            params_list.append([D, sigma])
            phi_fft_list.append(phi_fft)
            phi_corr_list.append(phi_corr)
    
    params_arr = np.array(params_list)
    phi_fft_arr = np.array(phi_fft_list)
    phi_corr_arr = np.array(phi_corr_list)
    
    print(f"Analyzed {len(params_arr)} samples")
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    # Normalize parameters
    params_norm = (params_arr - params_arr.mean(0)) / (params_arr.std(0) + 1e-8)
    param_dist = squareform(pdist(params_norm, metric='euclidean'))
    
    # Normalize Phi
    phi_fft_norm = (phi_fft_arr - phi_fft_arr.mean(0)) / (phi_fft_arr.std(0) + 1e-8)
    phi_corr_norm = (phi_corr_arr - phi_corr_arr.mean(0)) / (phi_corr_arr.std(0) + 1e-8)
    phi_combined = np.concatenate([phi_fft_norm, phi_corr_norm], axis=1)
    
    phi_fft_dist = squareform(pdist(phi_fft_norm, metric='euclidean'))
    phi_corr_dist = squareform(pdist(phi_corr_norm, metric='euclidean'))
    phi_combined_dist = squareform(pdist(phi_combined, metric='euclidean'))
    
    # Correlation between distances
    mask = np.triu(np.ones_like(param_dist, dtype=bool), k=1)
    
    corr_fft = np.corrcoef(param_dist[mask], phi_fft_dist[mask])[0, 1]
    corr_corr = np.corrcoef(param_dist[mask], phi_corr_dist[mask])[0, 1]
    corr_combined = np.corrcoef(param_dist[mask], phi_combined_dist[mask])[0, 1]
    
    print(f"\nDistance correlation (parameter space ↔ Phi space):")
    print(f"  FFT only:      {corr_fft:.4f}")
    print(f"  Corr only:     {corr_corr:.4f}")
    print(f"  Combined:      {corr_combined:.4f}")
    
    print("\n--- Assessment ---")
    if corr_combined < 0.3:
        print("⚠️  CRITICAL: Low correlation ({:.2f}) suggests (D,σ) is NOT identifiable from Φ!".format(corr_combined))
        print("    Different parameters produce similar physics signatures.")
        print("    Your inverse problem may be fundamentally ill-posed.")
    elif corr_combined < 0.5:
        print("⚠️  WARNING: Moderate correlation ({:.2f}). Expect noisy predictions.".format(corr_combined))
    else:
        print("✓  Good identifiability ({:.2f}): Φ can distinguish different (D,σ)".format(corr_combined))
    
    # Check for collisions
    collision_threshold_phi = np.percentile(phi_combined_dist[mask], 10)  # Bottom 10%
    collision_threshold_param = np.percentile(param_dist[mask], 50)  # Top 50%
    
    n_collisions = 0
    for i in range(len(params_arr)):
        for j in range(i+1, len(params_arr)):
            if phi_combined_dist[i,j] < collision_threshold_phi and param_dist[i,j] > collision_threshold_param:
                n_collisions += 1
    
    print(f"\n  Collision cases (similar Φ but different params): {n_collisions}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(param_dist[mask], phi_fft_dist[mask], alpha=0.1, s=1)
    axes[0].set_xlabel("Parameter distance")
    axes[0].set_ylabel("FFT Phi distance")
    axes[0].set_title(f"FFT: corr = {corr_fft:.3f}")
    
    axes[1].scatter(param_dist[mask], phi_corr_dist[mask], alpha=0.1, s=1)
    axes[1].set_xlabel("Parameter distance")
    axes[1].set_ylabel("Corr Phi distance")
    axes[1].set_title(f"Corr: corr = {corr_corr:.3f}")
    
    axes[2].scatter(param_dist[mask], phi_combined_dist[mask], alpha=0.1, s=1)
    axes[2].set_xlabel("Parameter distance")
    axes[2].set_ylabel("Combined Phi distance")
    axes[2].set_title(f"Combined: corr = {corr_combined:.3f}")
    
    plt.tight_layout()
    plt.savefig("identifiability_diagnostic.png", dpi=150)
    print("\nSaved: identifiability_diagnostic.png")
    
    return {
        "corr_fft": corr_fft,
        "corr_corr": corr_corr,
        "corr_combined": corr_combined,
        "n_collisions": n_collisions
    }


# ==============================================================================
# DIAGNOSTIC 3: Parameter Distribution Analysis
# ==============================================================================

def analyze_parameter_distribution():
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: PARAMETER DISTRIBUTION")
    print("="*60)
    
    df = pd.read_csv(CSV_PATH)
    
    D_values = df[COL_D].values * 1e3  # Convert to mJ/m²
    sigma_values = df[COL_SIGMA].values
    
    print(f"\nD (mJ/m²):")
    print(f"  Range: [{D_values.min():.3f}, {D_values.max():.3f}]")
    print(f"  Mean:  {D_values.mean():.3f}")
    print(f"  Std:   {D_values.std():.3f}")
    
    print(f"\nσ:")
    print(f"  Range: [{sigma_values.min():.4f}, {sigma_values.max():.4f}]")
    print(f"  Mean:  {sigma_values.mean():.4f}")
    print(f"  Std:   {sigma_values.std():.4f}")
    
    # Check for problematic regions
    low_D = (D_values < 0.3).sum()
    low_sigma = (sigma_values < 0.02).sum()
    
    print(f"\n--- Potential Problem Regions ---")
    print(f"  Samples with D < 0.3 mJ/m²: {low_D} ({100*low_D/len(D_values):.1f}%)")
    print(f"  Samples with σ < 0.02:      {low_sigma} ({100*low_sigma/len(sigma_values):.1f}%)")
    
    if low_D > 0.1 * len(D_values):
        print("  ⚠️  Many samples have low D - domain patterns may be degenerate")
    if low_sigma > 0.1 * len(sigma_values):
        print("  ⚠️  Many samples have low σ - stochastic variation dominates")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].hist(D_values, bins=50, edgecolor='black')
    axes[0].set_xlabel("D (mJ/m²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("D Distribution")
    axes[0].axvline(0.3, color='r', linestyle='--', label='Low D threshold')
    axes[0].legend()
    
    axes[1].hist(sigma_values, bins=50, edgecolor='black')
    axes[1].set_xlabel("σ")
    axes[1].set_ylabel("Count")
    axes[1].set_title("σ Distribution")
    axes[1].axvline(0.02, color='r', linestyle='--', label='Low σ threshold')
    axes[1].legend()
    
    axes[2].scatter(D_values, sigma_values, alpha=0.3, s=5)
    axes[2].set_xlabel("D (mJ/m²)")
    axes[2].set_ylabel("σ")
    axes[2].set_title("Parameter Space Coverage")
    
    plt.tight_layout()
    plt.savefig("parameter_distribution.png", dpi=150)
    print("\nSaved: parameter_distribution.png")
    
    return {
        "D_range": (D_values.min(), D_values.max()),
        "sigma_range": (sigma_values.min(), sigma_values.max()),
        "low_D_count": low_D,
        "low_sigma_count": low_sigma
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PINN FULL DIAGNOSTIC SUITE")
    print("="*60)
    
    # Check paths exist
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Please update the path in the script.")
        exit(1)
    
    if not IMG_DIR.exists():
        print(f"ERROR: Image directory not found at {IMG_DIR}")
        print("Please update the path in the script.")
        exit(1)
    
    # Run diagnostics
    print("\nRunning diagnostics...")
    
    # 1. Parameter distribution
    param_results = analyze_parameter_distribution()
    
    # 2. Identifiability
    ident_results = check_identifiability()
    
    # 3. Surrogate quality (if checkpoints exist)
    if SURR_FFT_PATH.exists() and SURR_CORR_PATH.exists():
        surr_results = check_surrogate_quality()
    else:
        print("\n⚠️  Surrogate checkpoints not found. Skipping surrogate quality check.")
        print(f"   Expected: {SURR_FFT_PATH}")
        print(f"   Expected: {SURR_CORR_PATH}")
        surr_results = None
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print("\n1. PARAMETER DISTRIBUTION:")
    print(f"   D range: {param_results['D_range'][0]:.3f} - {param_results['D_range'][1]:.3f} mJ/m²")
    print(f"   σ range: {param_results['sigma_range'][0]:.4f} - {param_results['sigma_range'][1]:.4f}")
    
    print("\n2. IDENTIFIABILITY:")
    print(f"   Distance correlation: {ident_results['corr_combined']:.3f}")
    if ident_results['corr_combined'] < 0.3:
        print("   ⚠️  POOR - inverse problem may be ill-posed")
    elif ident_results['corr_combined'] < 0.5:
        print("   ⚠️  MODERATE - expect noisy predictions")
    else:
        print("   ✓  GOOD")
    
    if surr_results:
        print("\n3. SURROGATE QUALITY:")
        print(f"   FFT R²:  {surr_results['r2_fft']:.3f}")
        print(f"   Corr R²: {surr_results['r2_corr']:.3f}")
        if surr_results['r2_fft'] < 0.5 or surr_results['r2_corr'] < 0.5:
            print("   ⚠️  POOR - physics loss may be harmful")
        elif surr_results['r2_fft'] < 0.8 or surr_results['r2_corr'] < 0.8:
            print("   ⚠️  MODERATE - use physics loss carefully")
        else:
            print("   ✓  GOOD")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    if ident_results['corr_combined'] < 0.5:
        recommendations.append("1. Consider adding more discriminative observables (domain wall width, anisotropic features)")
    
    if surr_results and (surr_results['r2_fft'] < 0.5 or surr_results['r2_corr'] < 0.5):
        recommendations.append("2. Train baseline CNN WITHOUT physics loss first (set lambda_phys=0)")
    
    if param_results['low_D_count'] > 0.1 * 2370:
        recommendations.append("3. Filter dataset to remove samples with D < 0.3 mJ/m²")
    
    if param_results['low_sigma_count'] > 0.1 * 2370:
        recommendations.append("4. Filter dataset to remove samples with σ < 0.02")
    
    recommendations.append("5. Use the fixed 16-bit image loading (MzDataset16bit)")
    recommendations.append("6. Try the attention-based CNN (DomainWallAttentionCNN)")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "="*60)
    print("Done! Check the saved PNG files for visualizations.")
    print("="*60)
#!/usr/bin/env python3
"""
Baseline CNN Training - NO Physics Loss
========================================
This script trains a pure supervised CNN to establish the baseline
performance without any physics-informed components.

This tells us what's achievable with just data supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# CONFIGURATION - UPDATE THESE
# ==============================================================================

CSV_PATH = Path("../mumax_dataset_ku_by_block_disorder_phy_corrected_3/training_index.csv")
IMG_DIR = Path("../images/corrected_3")

IMG_PREFIX = "mz_binary_"
IMG_EXT = ".png"
COL_D = "Dind"
COL_SIGMA = "sigma_nominal"

# Training config
BATCH_SIZE = 8  # Reduced for GPU memory
LR = 1e-4
EPOCHS = 100
VAL_RATIO = 0.15
RANDOM_SEED = 42

# Memory optimization
torch.backends.cudnn.benchmark = True

# Filter problematic samples?
FILTER_LOW_D = True      # Remove D < 0.3 mJ/m²
FILTER_LOW_SIGMA = True  # Remove σ < 0.02

D_MIN_THRESHOLD = 0.3e-3    # 0.3 mJ/m² in SI units
SIGMA_MIN_THRESHOLD = 0.02

OUTPUT_DIR = Path("./baseline_results_grad_corrected_V2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 16-BIT IMAGE LOADING (FIXED)
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
        print(f"WARNING: {path} loaded as 8-bit!")
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected image mode: {img.mode}")
    return arr


# ==============================================================================
# DATASET
# ==============================================================================

class MzDataset16bit(Dataset):
    def __init__(self, csv_path, img_dir, 
                 filter_low_D=False, D_min=0.3e-3,
                 filter_low_sigma=False, sigma_min=0.02,
                 target_mean=None, target_std=None):
        
        self.df = pd.read_csv(csv_path)
        self.df.set_index("run_id", inplace=True)
        self.img_dir = Path(img_dir)
        
        self.records = []
        n_filtered_D = 0
        n_filtered_sigma = 0
        
        img_files = sorted(self.img_dir.glob(f"{IMG_PREFIX}*{IMG_EXT}"))
        
        for img_path in img_files:
            try:
                run_id = int(img_path.stem.replace(IMG_PREFIX, ""))
            except ValueError:
                continue
            
            if run_id not in self.df.index:
                continue
            
            row = self.df.loc[run_id]
            D = float(row[COL_D])
            sigma = float(row[COL_SIGMA])
            
            # Apply filters
            if filter_low_D and D < D_min:
                n_filtered_D += 1
                continue
            if filter_low_sigma and sigma < sigma_min:
                n_filtered_sigma += 1
                continue
            
            self.records.append({
                "img_path": img_path,
                "D": D,
                "sigma": sigma,
                "run_id": run_id
            })
        
        print(f"[Dataset] Loaded {len(self.records)} samples")
        if filter_low_D:
            print(f"  Filtered {n_filtered_D} samples with D < {D_min*1e3:.1f} mJ/m²")
        if filter_low_sigma:
            print(f"  Filtered {n_filtered_sigma} samples with σ < {sigma_min}")
        
        self.target_mean = target_mean
        self.target_std = target_std
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        r = self.records[idx]
        
        # Load 16-bit image
        arr = load_16bit_image(r["img_path"])
        img = torch.from_numpy(arr).unsqueeze(0).float()
        img = img.repeat(3, 1, 1)  # 3 channels for CNN
        
        # Targets
        target = np.array([r["D"], r["sigma"]], dtype=np.float32)
        
        if self.target_mean is not None:
            target = (target - self.target_mean) / self.target_std
        
        target = torch.tensor(target, dtype=torch.float32)
        
        return img, target


# ==============================================================================
# CNN MODEL (with gradient channel option)
# ==============================================================================

class BaselineCNN(nn.Module):
    """Standard CNN for regression."""
    
    def __init__(self, use_gradient_channel=True):
        super().__init__()
        self.use_gradient_channel = use_gradient_channel
        
        in_ch = 4 if use_gradient_channel else 3
        
        # Sobel filters
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        
        self.features = nn.Sequential(
            conv_block(in_ch, 32),   # 512 -> 256
            conv_block(32, 64),      # 256 -> 128
            conv_block(64, 128),     # 128 -> 64
            conv_block(128, 256),    # 64 -> 32
            conv_block(256, 512),    # 32 -> 16
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
    
    def compute_gradient(self, x):
        gray = x[:, 0:1, :, :]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        print(grad)
        return grad / (grad.max() + 1e-8)
        #grad = grad / (grad.amax(dim=(2,3), keepdim=True) + 1e-8)

    
    def forward(self, x):
        if self.use_gradient_channel:
            grad = self.compute_gradient(x)
            x = torch.cat([x, grad], dim=1)
        
        x = self.features(x)
        x = self.global_pool(x)
        return self.regressor(x)


# ==============================================================================
# TRAINING
# ==============================================================================

def train_baseline():
    print("\n" + "="*60)
    print("BASELINE CNN TRAINING (NO PHYSICS LOSS)")
    print("="*60)
    
    # Create dataset
    full_dataset = MzDataset16bit(
        CSV_PATH, IMG_DIR,
        filter_low_D=FILTER_LOW_D,
        D_min=D_MIN_THRESHOLD,
        filter_low_sigma=FILTER_LOW_SIGMA,
        sigma_min=SIGMA_MIN_THRESHOLD
    )
    
    # Split
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    print(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Compute normalization from training set
    train_targets = np.array([
        [full_dataset.records[i]["D"], full_dataset.records[i]["sigma"]]
        for i in train_dataset.indices
    ], dtype=np.float32)
    
    t_mean = train_targets.mean(axis=0)
    t_std = train_targets.std(axis=0)
    t_std[t_std == 0] = 1.0
    
    print(f"Target normalization:")
    print(f"  D:     mean={t_mean[0]*1e3:.3f} mJ/m², std={t_std[0]*1e3:.3f}")
    print(f"  sigma: mean={t_mean[1]:.4f}, std={t_std[1]:.4f}")
    
    # Apply normalization
    full_dataset.target_mean = t_mean
    full_dataset.target_std = t_std
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = BaselineCNN(use_gradient_channel=True).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()
    
    # Mixed precision for memory efficiency
    #scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    use_amp = True
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_mae_D': [], 'val_mae_sigma': [],
        'val_mape_D': [], 'val_mape_sigma': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("-" * 70)
    
    # Clear GPU memory before training
    torch.cuda.empty_cache()
    
    for epoch in range(1, EPOCHS + 1):
        # ==================== TRAIN ====================
        model.train()
        train_loss = 0.0
        
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            #with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # ==================== VALIDATE ====================
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Unnormalize for metrics
                preds_unnorm = outputs.cpu().numpy() * t_std + t_mean
                targets_unnorm = targets.cpu().numpy() * t_std + t_mean
                
                all_preds.append(preds_unnorm)
                all_targets.append(targets_unnorm)
        
        val_loss /= len(val_loader)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mae_D = np.abs(all_preds[:, 0] - all_targets[:, 0]).mean()
        mae_sigma = np.abs(all_preds[:, 1] - all_targets[:, 1]).mean()
        
        mape_D = (np.abs(all_preds[:, 0] - all_targets[:, 0]) / (np.abs(all_targets[:, 0]) + 1e-10)).mean() * 100
        mape_sigma = (np.abs(all_preds[:, 1] - all_targets[:, 1]) / (np.abs(all_targets[:, 1]) + 1e-10)).mean() * 100
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae_D'].append(mae_D)
        history['val_mae_sigma'].append(mae_sigma)
        history['val_mape_D'].append(mape_D)
        history['val_mape_sigma'].append(mape_sigma)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4e} | Val: {val_loss:.4e} | "
                  f"MAE D: {mae_D*1e3:.3f} mJ/m² | MAE σ: {mae_sigma:.4f} | "
                  f"MAPE D: {mape_D:.1f}% | MAPE σ: {mape_sigma:.1f}% | LR: {lr_now:.1e}")
    
    print("-" * 70)
    print("Training complete!")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Save model
    model_path = OUTPUT_DIR / "baseline_model.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'target_mean': t_mean,
        'target_std': t_std,
        'history': history,
        'config': {
            'filter_low_D': FILTER_LOW_D,
            'filter_low_sigma': FILTER_LOW_SIGMA,
            'D_min': D_MIN_THRESHOLD,
            'sigma_min': SIGMA_MIN_THRESHOLD
        }
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs)
            
            preds_unnorm = outputs.cpu().numpy() * t_std + t_mean
            targets_unnorm = targets.cpu().numpy() * t_std + t_mean
            
            all_preds.append(preds_unnorm)
            all_targets.append(targets_unnorm)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # D metrics
    D_pred = all_preds[:, 0] * 1e3  # Convert to mJ/m²
    D_true = all_targets[:, 0] * 1e3
    
    mae_D = np.abs(D_pred - D_true).mean()
    rmse_D = np.sqrt(((D_pred - D_true) ** 2).mean())
    mape_D = (np.abs(D_pred - D_true) / (np.abs(D_true) + 1e-10)).mean() * 100
    corr_D = np.corrcoef(D_pred, D_true)[0, 1]
    r2_D = 1 - ((D_pred - D_true) ** 2).sum() / ((D_true - D_true.mean()) ** 2).sum()
    
    print(f"\nD (mJ/m²):")
    print(f"  MAE:  {mae_D:.4f}")
    print(f"  RMSE: {rmse_D:.4f}")
    print(f"  MAPE: {mape_D:.1f}%")
    print(f"  Corr: {corr_D:.4f}")
    print(f"  R²:   {r2_D:.4f}")
    
    # Sigma metrics
    sigma_pred = all_preds[:, 1]
    sigma_true = all_targets[:, 1]
    
    mae_sigma = np.abs(sigma_pred - sigma_true).mean()
    rmse_sigma = np.sqrt(((sigma_pred - sigma_true) ** 2).mean())
    mape_sigma = (np.abs(sigma_pred - sigma_true) / (np.abs(sigma_true) + 1e-10)).mean() * 100
    corr_sigma = np.corrcoef(sigma_pred, sigma_true)[0, 1]
    r2_sigma = 1 - ((sigma_pred - sigma_true) ** 2).sum() / ((sigma_true - sigma_true.mean()) ** 2).sum()
    
    print(f"\nσ:")
    print(f"  MAE:  {mae_sigma:.4f}")
    print(f"  RMSE: {rmse_sigma:.4f}")
    print(f"  MAPE: {mape_sigma:.1f}%")
    print(f"  Corr: {corr_sigma:.4f}")
    print(f"  R²:   {r2_sigma:.4f}")
    
    # ==================== PLOTS ====================
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['val_mape_D'], label='D')
    axes[0, 1].plot(history['val_mape_sigma'], label='σ')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAPE (%)')
    axes[0, 1].set_title('Validation MAPE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 2. Prediction scatter plots
    axes[1, 0].scatter(D_true, D_pred, alpha=0.5, s=10)
    axes[1, 0].plot([D_true.min(), D_true.max()], [D_true.min(), D_true.max()], 'r--', label='Perfect')
    axes[1, 0].set_xlabel('True D (mJ/m²)')
    axes[1, 0].set_ylabel('Predicted D (mJ/m²)')
    axes[1, 0].set_title(f'D Prediction (R²={r2_D:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].scatter(sigma_true, sigma_pred, alpha=0.5, s=10)
    axes[1, 1].plot([sigma_true.min(), sigma_true.max()], [sigma_true.min(), sigma_true.max()], 'r--', label='Perfect')
    axes[1, 1].set_xlabel('True σ')
    axes[1, 1].set_ylabel('Predicted σ')
    axes[1, 1].set_title(f'σ Prediction (R²={r2_sigma:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "baseline_results.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nPlot saved to {fig_path}")
    plt.show()
    
    # 3. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(D_pred - D_true, bins=50, edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--')
    axes[0].set_xlabel('D Error (mJ/m²)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'D Error Distribution (MAE={mae_D:.3f})')
    
    axes[1].hist(sigma_pred - sigma_true, bins=50, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('σ Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'σ Error Distribution (MAE={mae_sigma:.4f})')
    
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "baseline_error_distribution.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Error distribution saved to {fig_path}")
    
    return {
        'D': {'mae': mae_D, 'rmse': rmse_D, 'mape': mape_D, 'corr': corr_D, 'r2': r2_D},
        'sigma': {'mae': mae_sigma, 'rmse': rmse_sigma, 'mape': mape_sigma, 'corr': corr_sigma, 'r2': r2_sigma}
    }


if __name__ == "__main__":
    results = train_baseline()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBaseline CNN (no physics loss) achieves:")
    print(f"  D:     R²={results['D']['r2']:.3f}, MAPE={results['D']['mape']:.1f}%")
    print(f"  sigma: R²={results['sigma']['r2']:.3f}, MAPE={results['sigma']['mape']:.1f}%")
    print("\nIf MAPE < 10-15%, baseline is already good!")
    print("Physics loss should only be added if it IMPROVES on this baseline.")
#!/usr/bin/env python3
"""
================================================================================
IMPROVED PHYSICS-INFORMED NEURAL NETWORK FOR MICROMAGNETIC PARAMETER PREDICTION
================================================================================

Key Improvements over Original:
1. Corrected energy loss with nonlinear D-dependence
2. Proper spectral-based σ observable with theoretical foundation
3. Gradient conflict detection and adaptive loss weighting
4. Per-parameter physics (separate losses for D and σ)
5. OOD evaluation framework
6. Loss annealing schedule

Author: [Your Name]
Date: 2025
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==============================================================================
# PHYSICAL CONSTANTS AND CONFIGURATION
# ==============================================================================

@dataclass
class PhysicsConfig:
    """Physical parameters for micromagnetic simulations."""
    Aex: float = 3.1e-11      # Exchange stiffness [J/m]
    Ms: float = 1.445e6       # Saturation magnetization [A/m]
    Ku: float = 1.6e6         # Base anisotropy [J/m³]
    cell_x: float = 4e-9      # Cell size x [m]
    cell_y: float = 4e-9      # Cell size y [m]
    thickness: float = 0.9e-9  # Film thickness [m]
    
    @property
    def exchange_length(self) -> float:
        """Bloch wall parameter: l_ex = √(A/K)"""
        return np.sqrt(self.Aex / self.Ku)
    
    @property
    def quality_factor(self) -> float:
        """Material quality factor Q = √(A·K)"""
        return np.sqrt(self.Aex * self.Ku)
    
    @property
    def critical_dmi(self) -> float:
        """Critical DMI for transition: D_c = 4√(AK)/π"""
        return 4 * np.sqrt(self.Aex * self.Ku) / np.pi
    
    @property
    def domain_wall_width(self) -> float:
        """Bloch wall width: δ = π√(A/K)"""
        return np.pi * self.exchange_length


# ==============================================================================
# IMPROVED ENERGY COMPUTATION WITH PROPER PHYSICS
# ==============================================================================

def compute_spatial_gradients(field: torch.Tensor, dx: float, dy: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial gradients using central differences with periodic BC.
    
    Args:
        field: [B, H, W] tensor
        dx, dy: cell sizes in meters
    
    Returns:
        grad_x, grad_y: [B, H, W] tensors
    """
    grad_x = (torch.roll(field, -1, dims=2) - torch.roll(field, 1, dims=2)) / (2 * dx)
    grad_y = (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / (2 * dy)
    return grad_x, grad_y


def compute_exchange_energy_density(mx, my, mz, Aex, dx, dy) -> torch.Tensor:
    """
    Exchange energy density: e_ex = A·(∇m)²
    
    Returns: [B] tensor of mean exchange energy density per sample
    """
    dmx_dx, dmx_dy = compute_spatial_gradients(mx, dx, dy)
    dmy_dx, dmy_dy = compute_spatial_gradients(my, dx, dy)
    dmz_dx, dmz_dy = compute_spatial_gradients(mz, dx, dy)
    
    grad_m_squared = (
        dmx_dx**2 + dmx_dy**2 +
        dmy_dx**2 + dmy_dy**2 +
        dmz_dx**2 + dmz_dy**2
    )
    
    e_ex = Aex * grad_m_squared
    return e_ex.mean(dim=(1, 2))


def compute_dmi_energy_density(mx, my, mz, D, dx, dy) -> torch.Tensor:
    """
    Interfacial DMI energy density (Néel type).
    
    e_DMI = D·[mz·(∂mx/∂x) - mx·(∂mz/∂x) + mz·(∂my/∂y) - my·(∂mz/∂y)]
    
    Returns: [B] tensor of mean DMI energy density per sample
    """
    dmx_dx, _ = compute_spatial_gradients(mx, dx, dy)
    _, dmy_dy = compute_spatial_gradients(my, dx, dy)
    dmz_dx, dmz_dy = compute_spatial_gradients(mz, dx, dy)
    
    dmi_term = (
        mz * dmx_dx - mx * dmz_dx +
        mz * dmy_dy - my * dmz_dy
    )
    
    if isinstance(D, torch.Tensor) and D.dim() > 0:
        D = D.view(-1, 1, 1)
    
    e_dmi = D * dmi_term
    return e_dmi.mean(dim=(1, 2))


def compute_total_gradient_magnitude(mz, dx, dy) -> torch.Tensor:
    """
    Compute total gradient magnitude of mz field.
    Related to domain wall density.
    
    Returns: [B] tensor
    """
    dmz_dx, dmz_dy = compute_spatial_gradients(mz, dx, dy)
    grad_mag = torch.sqrt(dmz_dx**2 + dmz_dy**2 + 1e-10)
    return grad_mag.mean(dim=(1, 2))


# ==============================================================================
# IMPROVED PHYSICS LOSS WITH CORRECT THEORY
# ==============================================================================

class ImprovedPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss with:
    1. Corrected nonlinear energy ratio for D
    2. Spectral width observable for σ
    3. Gradient conflict detection
    4. Adaptive weighting support
    """
    
    def __init__(self, 
                 phys_config: PhysicsConfig,
                 target_mean: np.ndarray,
                 target_std: np.ndarray,
                 loss_weights: Dict[str, float],
                 use_adaptive_weights: bool = False):
        super().__init__()
        
        self.config = phys_config
        self.register_buffer('target_mean', torch.tensor(target_mean, dtype=torch.float32))
        self.register_buffer('target_std', torch.tensor(target_std, dtype=torch.float32))
        
        self.base_weights = loss_weights
        self.use_adaptive = use_adaptive_weights
        
        # Critical DMI as tensor
        self.register_buffer('D_critical', 
                            torch.tensor(phys_config.critical_dmi, dtype=torch.float32))
        self.register_buffer('Q', 
                            torch.tensor(phys_config.quality_factor, dtype=torch.float32))
        
        # For adaptive weighting
        self.grad_history = {
            'regression': [],
            'energy': [],
            'spectral': []
        }
        
    def denormalize_params(self, params_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions to physical units."""
        return params_norm * self.target_std + self.target_mean
    
    def compute_energy_loss_improved(self, 
                                      D_pred: torch.Tensor,
                                      mx: torch.Tensor,
                                      my: torch.Tensor,
                                      mz: torch.Tensor) -> torch.Tensor:
        """
        IMPROVED energy ratio loss with nonlinear D-dependence.
        
        Physics:
        - E_DMI/E_ex ∝ D·δ/A where δ is domain wall width
        - For labyrinthine domains: δ ∝ √(A/K)
        - Therefore: E_DMI/E_ex ∝ D/√(A·K) = D/Q
        
        But this saturates near D_critical = 4√(AK)/π
        
        Improved formula:
        ratio_theory = (D/D_c) / (1 + (D/D_c)²)^0.5
        
        This captures:
        1. Linear behavior at low D
        2. Saturation at high D
        3. Physical bounds
        """
        device = D_pred.device
        
        # Compute actual energies
        E_ex = compute_exchange_energy_density(
            mx, my, mz, self.config.Aex, self.config.cell_x, self.config.cell_y
        )
        E_dmi = compute_dmi_energy_density(
            mx, my, mz, D_pred, self.config.cell_x, self.config.cell_y
        )
        
        # Measured ratio (dimensionless)
        ratio_measured = torch.abs(E_dmi) / (torch.abs(E_ex) + 1e-10)
        
        # IMPROVED theoretical ratio with saturation
        d_normalized = D_pred / self.D_critical.to(device)
        ratio_theory = d_normalized / torch.sqrt(1 + d_normalized**2 + 1e-8)
        
        # Scale both to have similar magnitude (match means)
        ratio_measured_scaled = ratio_measured / (ratio_measured.mean() + 1e-8)
        ratio_theory_scaled = ratio_theory / (ratio_theory.mean() + 1e-8)
        
        # MSE loss
        loss = F.mse_loss(ratio_measured_scaled, ratio_theory_scaled)
        
        return loss, {
            'ratio_measured': ratio_measured.detach(),
            'ratio_theory': ratio_theory.detach(),
            'E_ex': E_ex.detach(),
            'E_dmi': E_dmi.detach()
        }
    
    def compute_spectral_loss_improved(self,
                                        sigma_pred: torch.Tensor,
                                        imgs: torch.Tensor) -> torch.Tensor:
        """
        IMPROVED spectral loss based on Fourier analysis.
        
        Physics:
        - Higher disorder (σ) → more irregular domain patterns
        - Irregular patterns → broader Fourier spectrum
        - Spectral width ∝ disorder level
        
        Observable: Spectral width (second moment of power spectrum)
        Theory: σ_spectral = σ_0 · (1 + β·σ_disorder)
        
        where σ_0 is base spectral width for ordered system
        and β is a coupling constant (~1-10, can be fitted)
        """
        device = imgs.device
        
        # Extract mz channel and convert to [-1, 1]
        mz_img = imgs[:, 0, :, :]
        mz_centered = 2.0 * mz_img - 1.0
        
        B, H, W = mz_centered.shape
        
        # Remove DC component
        mz_centered = mz_centered - mz_centered.mean(dim=(1, 2), keepdim=True)
        
        # FFT and power spectrum
        fft = torch.fft.fft2(mz_centered)
        fft_shifted = torch.fft.fftshift(fft, dim=(1, 2))
        power = torch.abs(fft_shifted) ** 2
        
        # Normalize power spectrum
        power_norm = power / (power.sum(dim=(1, 2), keepdim=True) + 1e-10)
        
        # Create radial coordinate grid
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        k_radial = torch.sqrt(xx**2 + yy**2)
        
        # Compute spectral moments
        # First moment (spectral centroid)
        k_mean = (power_norm * k_radial).sum(dim=(1, 2))
        
        # Second moment (spectral width/variance)
        k_var = (power_norm * (k_radial - k_mean.view(-1, 1, 1))**2).sum(dim=(1, 2))
        spectral_width = torch.sqrt(k_var + 1e-10)
        
        # Theoretical relationship:
        # Spectral width increases with disorder
        # Using linear relationship: width ∝ (1 + α·σ)
        sigma_normalized = sigma_pred / 0.15  # Normalize to [0, 1] range roughly
        
        # Base spectral width (for σ=0, perfectly ordered)
        # This should be estimated from data or set as hyperparameter
        sigma_0 = 0.2  # typical base width
        alpha = 2.0    # coupling strength (tunable)
        
        width_theory = sigma_0 * (1 + alpha * sigma_normalized)
        
        # Normalize both for comparison
        spectral_width_norm = spectral_width / (spectral_width.mean() + 1e-8)
        width_theory_norm = width_theory / (width_theory.mean() + 1e-8)
        
        loss = F.mse_loss(spectral_width_norm, width_theory_norm)
        
        return loss, {
            'spectral_width': spectral_width.detach(),
            'width_theory': width_theory.detach(),
            'k_mean': k_mean.detach()
        }
    
    def compute_gradient_magnitude_loss(self,
                                         D_pred: torch.Tensor,
                                         mz: torch.Tensor) -> torch.Tensor:
        """
        Alternative D-physics: Total gradient magnitude relates to domain wall density.
        
        Physics:
        - Higher D → more domain walls → higher total |∇mz|
        - |∇mz|_total ∝ D / √(A·K) for labyrinthine domains
        """
        device = D_pred.device
        
        grad_mag = compute_total_gradient_magnitude(
            mz, self.config.cell_x, self.config.cell_y
        )
        
        # Theory: gradient magnitude increases with D
        d_normalized = D_pred / self.D_critical.to(device)
        grad_theory = d_normalized / torch.sqrt(1 + d_normalized**2 + 1e-8)
        
        # Normalize
        grad_mag_norm = grad_mag / (grad_mag.mean() + 1e-8)
        grad_theory_norm = grad_theory / (grad_theory.mean() + 1e-8)
        
        loss = F.mse_loss(grad_mag_norm, grad_theory_norm)
        
        return loss, {
            'grad_mag': grad_mag.detach(),
            'grad_theory': grad_theory.detach()
        }
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                m_fields: Optional[Dict[str, torch.Tensor]],
                imgs: torch.Tensor,
                return_components: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total physics-informed loss.
        
        Returns:
            total_loss: weighted sum of all components
            loss_dict: individual loss values and diagnostics
        """
        device = predictions.device
        
        # Denormalize to physical units
        params_pred = self.denormalize_params(predictions)
        D_pred = params_pred[:, 0]
        sigma_pred = params_pred[:, 1]
        
        # ==========================================
        # 1. Regression Loss
        # ==========================================
        loss_reg = F.mse_loss(predictions, targets)
        
        # ==========================================
        # 2. Energy Loss (for D)
        # ==========================================
        loss_energy = torch.tensor(0.0, device=device)
        energy_diagnostics = {}
        
        if m_fields is not None and self.base_weights.get('energy_ratio', 0) > 0:
            mx = m_fields['mx'].to(device)
            my = m_fields['my'].to(device)
            mz = m_fields['mz'].to(device)
            
            loss_energy, energy_diagnostics = self.compute_energy_loss_improved(
                D_pred, mx, my, mz
            )
        
        # ==========================================
        # 3. Spectral Loss (for σ)
        # ==========================================
        loss_spectral = torch.tensor(0.0, device=device)
        spectral_diagnostics = {}
        
        if self.base_weights.get('spectral', 0) > 0:
            loss_spectral, spectral_diagnostics = self.compute_spectral_loss_improved(
                sigma_pred, imgs
            )
        
        # ==========================================
        # 4. Gradient Magnitude Loss (alternative for D)
        # ==========================================
        loss_grad_mag = torch.tensor(0.0, device=device)
        grad_diagnostics = {}
        
        if m_fields is not None and self.base_weights.get('grad_magnitude', 0) > 0:
            mz = m_fields['mz'].to(device)
            loss_grad_mag, grad_diagnostics = self.compute_gradient_magnitude_loss(
                D_pred, mz
            )
        
        # ==========================================
        # 5. Combine Losses
        # ==========================================
        weights = self.base_weights
        
        total_loss = (
            weights.get('regression', 1.0) * loss_reg +
            weights.get('energy_ratio', 0.0) * loss_energy +
            weights.get('spectral', 0.0) * loss_spectral +
            weights.get('grad_magnitude', 0.0) * loss_grad_mag
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'regression': loss_reg.item(),
            'energy': loss_energy.item() if isinstance(loss_energy, torch.Tensor) else 0.0,
            'spectral': loss_spectral.item() if isinstance(loss_spectral, torch.Tensor) else 0.0,
            'grad_magnitude': loss_grad_mag.item() if isinstance(loss_grad_mag, torch.Tensor) else 0.0,
        }
        
        if return_components:
            loss_dict['components'] = {
                'loss_reg': loss_reg,
                'loss_energy': loss_energy,
                'loss_spectral': loss_spectral,
                'loss_grad_mag': loss_grad_mag
            }
            loss_dict['diagnostics'] = {
                'energy': energy_diagnostics,
                'spectral': spectral_diagnostics,
                'grad_mag': grad_diagnostics
            }
        
        return total_loss, loss_dict


# ==============================================================================
# GRADIENT CONFLICT DETECTOR
# ==============================================================================

class GradientConflictDetector:
    """
    Detects when physics losses have gradients pointing in opposite directions.
    
    Key insight: Two losses can both have large gradient magnitudes but 
    push in opposite directions, canceling each other out.
    
    This detector measures:
    1. Gradient magnitudes (as before)
    2. Gradient cosine similarity between loss components
    3. Effective gradient after combination
    """
    
    def __init__(self):
        self.history = {
            # Magnitudes
            'grad_norm_reg': [],
            'grad_norm_energy': [],
            'grad_norm_spectral': [],
            
            # Cosine similarities (conflict detection)
            'cosine_reg_energy': [],
            'cosine_reg_spectral': [],
            'cosine_energy_spectral': [],
            
            # Effective gradient after combination
            'grad_norm_combined': [],
            
            # Per-parameter gradients
            'grad_norm_D_head': [],
            'grad_norm_sigma_head': [],
        }
    
    def get_gradient_vector(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Flatten all gradients into a single vector."""
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        
        if len(grads) == 0:
            return None
        return torch.cat(grads)
    
    def compute_cosine_similarity(self, 
                                   grad1: Optional[torch.Tensor], 
                                   grad2: Optional[torch.Tensor]) -> float:
        """Compute cosine similarity between two gradient vectors."""
        if grad1 is None or grad2 is None:
            return 0.0
        
        norm1 = grad1.norm() + 1e-10
        norm2 = grad2.norm() + 1e-10
        
        cosine = (grad1 @ grad2) / (norm1 * norm2)
        return cosine.item()
    
    def track_gradients(self,
                        model: nn.Module,
                        loss_components: Dict[str, torch.Tensor],
                        loss_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Track gradient magnitudes and conflicts.
        
        Args:
            model: the neural network
            loss_components: dict with 'reg', 'energy', 'spectral' losses
            loss_weights: weights for each component
        
        Returns:
            dict with gradient statistics
        """
        stats = {}
        gradient_vectors = {}
        
        # Compute gradient for each loss component separately
        for name, loss in loss_components.items():
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                model.zero_grad()
                loss.backward(retain_graph=True)
                
                grad_vec = self.get_gradient_vector(model)
                if grad_vec is not None:
                    gradient_vectors[name] = grad_vec.clone()
                    stats[f'grad_norm_{name}'] = grad_vec.norm().item()
                else:
                    stats[f'grad_norm_{name}'] = 0.0
            else:
                stats[f'grad_norm_{name}'] = 0.0
        
        # Compute pairwise cosine similarities
        pairs = [('reg', 'energy'), ('reg', 'spectral'), ('energy', 'spectral')]
        for name1, name2 in pairs:
            if name1 in gradient_vectors and name2 in gradient_vectors:
                cosine = self.compute_cosine_similarity(
                    gradient_vectors[name1], gradient_vectors[name2]
                )
                stats[f'cosine_{name1}_{name2}'] = cosine
            else:
                stats[f'cosine_{name1}_{name2}'] = 0.0
        
        # Compute combined gradient norm
        model.zero_grad()
        total_loss = sum(
            loss_weights.get(name.replace('loss_', ''), 0.0) * loss
            for name, loss in loss_components.items()
            if isinstance(loss, torch.Tensor) and loss.requires_grad
        )
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward(retain_graph=True)
            combined_grad = self.get_gradient_vector(model)
            if combined_grad is not None:
                stats['grad_norm_combined'] = combined_grad.norm().item()
        
        # Record to history
        self.history['grad_norm_reg'].append(stats.get('grad_norm_reg', 0.0))
        self.history['grad_norm_energy'].append(stats.get('grad_norm_energy', 0.0))
        self.history['grad_norm_spectral'].append(stats.get('grad_norm_spectral', 0.0))
        self.history['cosine_reg_energy'].append(stats.get('cosine_reg_energy', 0.0))
        self.history['cosine_reg_spectral'].append(stats.get('cosine_reg_spectral', 0.0))
        self.history['cosine_energy_spectral'].append(stats.get('cosine_energy_spectral', 0.0))
        self.history['grad_norm_combined'].append(stats.get('grad_norm_combined', 0.0))
        
        return stats
    
    def get_conflict_summary(self) -> Dict[str, float]:
        """Get summary of gradient conflicts."""
        if len(self.history['cosine_reg_energy']) == 0:
            return {}
        
        return {
            'mean_cosine_reg_energy': np.mean(self.history['cosine_reg_energy']),
            'mean_cosine_reg_spectral': np.mean(self.history['cosine_reg_spectral']),
            'mean_cosine_energy_spectral': np.mean(self.history['cosine_energy_spectral']),
            'conflict_rate_reg_energy': np.mean(np.array(self.history['cosine_reg_energy']) < 0),
            'conflict_rate_reg_spectral': np.mean(np.array(self.history['cosine_reg_spectral']) < 0),
        }


# ==============================================================================
# ADAPTIVE LOSS WEIGHTING (GradNorm-inspired)
# ==============================================================================

class AdaptiveLossWeighter:
    """
    Dynamically adjust loss weights to balance gradient magnitudes.
    
    Inspired by GradNorm (Chen et al., 2018).
    
    Goal: Ensure all loss components contribute roughly equally to training,
    preventing any single loss from dominating.
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 alpha: float = 1.5,
                 update_rate: float = 0.01):
        """
        Args:
            initial_weights: starting weights for each loss
            alpha: strength of rebalancing (higher = more aggressive)
            update_rate: how quickly weights adapt
        """
        self.weights = initial_weights.copy()
        self.alpha = alpha
        self.update_rate = update_rate
        
        # Track initial loss values for relative progress
        self.initial_losses = None
        self.weight_history = {k: [v] for k, v in initial_weights.items()}
    
    def update_weights(self, 
                       current_losses: Dict[str, float],
                       grad_norms: Dict[str, float]) -> Dict[str, float]:
        """
        Update weights based on gradient magnitudes and loss progress.
        
        Args:
            current_losses: current value of each loss component
            grad_norms: gradient norm for each loss component
        
        Returns:
            updated weights
        """
        if self.initial_losses is None:
            self.initial_losses = current_losses.copy()
        
        # Compute relative inverse training rate (how much each task has improved)
        relative_losses = {}
        for name in current_losses:
            if name in self.initial_losses and self.initial_losses[name] > 0:
                relative_losses[name] = current_losses[name] / (self.initial_losses[name] + 1e-10)
            else:
                relative_losses[name] = 1.0
        
        # Compute mean relative loss
        mean_relative = np.mean(list(relative_losses.values()))
        
        # Compute target gradient ratios (tasks that haven't improved should get more weight)
        target_ratios = {}
        for name in relative_losses:
            target_ratios[name] = (relative_losses[name] / (mean_relative + 1e-10)) ** self.alpha
        
        # Normalize target ratios
        total_target = sum(target_ratios.values())
        for name in target_ratios:
            target_ratios[name] /= (total_target + 1e-10)
        
        # Update weights towards target
        for name in self.weights:
            if name in target_ratios:
                target_weight = target_ratios[name] * sum(self.weights.values())
                self.weights[name] += self.update_rate * (target_weight - self.weights[name])
        
        # Ensure weights are positive
        for name in self.weights:
            self.weights[name] = max(self.weights[name], 0.01)
        
        # Record history
        for name, weight in self.weights.items():
            self.weight_history[name].append(weight)
        
        return self.weights.copy()


# ==============================================================================
# LOSS ANNEALING SCHEDULE
# ==============================================================================

class LossAnnealingSchedule:
    """
    Gradually increase physics loss weights during training.
    
    Rationale:
    - Early training: focus on data fitting (high regression weight)
    - Late training: add physics constraints (increase physics weights)
    
    This prevents physics from interfering with initial learning.
    """
    
    def __init__(self,
                 final_weights: Dict[str, float],
                 warmup_epochs: int = 20,
                 schedule: str = 'linear'):
        """
        Args:
            final_weights: weights at end of training
            warmup_epochs: epochs before physics kicks in
            schedule: 'linear', 'cosine', or 'step'
        """
        self.final_weights = final_weights
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
    
    def get_weights(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Get weights for current epoch."""
        weights = {}
        
        # Regression always at full weight
        weights['regression'] = self.final_weights.get('regression', 1.0)
        
        # Physics weights are annealed
        physics_keys = ['energy_ratio', 'spectral', 'grad_magnitude']
        
        if epoch < self.warmup_epochs:
            # Warmup period: no physics
            physics_factor = 0.0
        else:
            # Annealing period
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs + 1e-10)
            progress = min(progress, 1.0)
            
            if self.schedule == 'linear':
                physics_factor = progress
            elif self.schedule == 'cosine':
                physics_factor = 0.5 * (1 - np.cos(np.pi * progress))
            elif self.schedule == 'step':
                physics_factor = 1.0 if progress > 0.5 else 0.0
            else:
                physics_factor = progress
        
        for key in physics_keys:
            weights[key] = self.final_weights.get(key, 0.0) * physics_factor
        
        return weights


# ==============================================================================
# OOD EVALUATION FRAMEWORK
# ==============================================================================

class OODEvaluator:
    """
    Evaluate model performance on out-of-distribution data.
    
    Compares:
    1. In-distribution (ID) performance
    2. Out-of-distribution (OOD) performance
    3. Physics consistency metrics
    """
    
    def __init__(self, 
                 phys_config: PhysicsConfig,
                 id_D_range: Tuple[float, float],
                 id_sigma_range: Tuple[float, float]):
        """
        Args:
            phys_config: physical parameters
            id_D_range: (D_min, D_max) for training data
            id_sigma_range: (σ_min, σ_max) for training data
        """
        self.config = phys_config
        self.id_D_range = id_D_range
        self.id_sigma_range = id_sigma_range
    
    def classify_sample(self, D: float, sigma: float) -> str:
        """Classify a sample as ID, OOD-D, OOD-sigma, or OOD-both."""
        d_in = self.id_D_range[0] <= D <= self.id_D_range[1]
        s_in = self.id_sigma_range[0] <= sigma <= self.id_sigma_range[1]
        
        if d_in and s_in:
            return 'ID'
        elif not d_in and s_in:
            return 'OOD-D'
        elif d_in and not s_in:
            return 'OOD-sigma'
        else:
            return 'OOD-both'
    
    def evaluate(self,
                 model: nn.Module,
                 dataloader,
                 target_mean: torch.Tensor,
                 target_std: torch.Tensor,
                 device: torch.device) -> Dict:
        """
        Evaluate model and return metrics by category.
        
        Returns:
            dict with metrics for ID, OOD-D, OOD-sigma, OOD-both
        """
        model.eval()
        
        results = {
            'ID': {'D_pred': [], 'D_true': [], 'sigma_pred': [], 'sigma_true': []},
            'OOD-D': {'D_pred': [], 'D_true': [], 'sigma_pred': [], 'sigma_true': []},
            'OOD-sigma': {'D_pred': [], 'D_true': [], 'sigma_pred': [], 'sigma_true': []},
            'OOD-both': {'D_pred': [], 'D_true': [], 'sigma_pred': [], 'sigma_true': []},
        }
        
        with torch.no_grad():
            for imgs, m_fields, targets, run_ids in dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                outputs = model(imgs)
                
                # Denormalize
                preds = outputs * target_std.to(device) + target_mean.to(device)
                trues = targets * target_std.to(device) + target_mean.to(device)
                
                for i in range(len(preds)):
                    D_true = trues[i, 0].item()
                    sigma_true = trues[i, 1].item()
                    
                    category = self.classify_sample(D_true, sigma_true)
                    
                    results[category]['D_pred'].append(preds[i, 0].item())
                    results[category]['D_true'].append(D_true)
                    results[category]['sigma_pred'].append(preds[i, 1].item())
                    results[category]['sigma_true'].append(sigma_true)
        
        # Compute metrics for each category
        metrics = {}
        for cat in results:
            if len(results[cat]['D_pred']) > 0:
                D_pred = np.array(results[cat]['D_pred'])
                D_true = np.array(results[cat]['D_true'])
                s_pred = np.array(results[cat]['sigma_pred'])
                s_true = np.array(results[cat]['sigma_true'])
                
                metrics[cat] = {
                    'n_samples': len(D_pred),
                    'D_MAE': np.mean(np.abs(D_pred - D_true)),
                    'D_MAPE': np.mean(np.abs((D_pred - D_true) / (D_true + 1e-10))) * 100,
                    'D_R2': 1 - np.sum((D_pred - D_true)**2) / (np.sum((D_true - D_true.mean())**2) + 1e-10),
                    'sigma_MAE': np.mean(np.abs(s_pred - s_true)),
                    'sigma_MAPE': np.mean(np.abs((s_pred - s_true) / (s_true + 1e-10))) * 100,
                    'sigma_R2': 1 - np.sum((s_pred - s_true)**2) / (np.sum((s_true - s_true.mean())**2) + 1e-10),
                }
            else:
                metrics[cat] = {'n_samples': 0}
        
        return metrics, results
    
    def plot_ood_comparison(self, 
                            metrics: Dict,
                            results: Dict,
                            save_path: Optional[Path] = None):
        """Create visualization of ID vs OOD performance."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        colors = {'ID': 'blue', 'OOD-D': 'red', 'OOD-sigma': 'green', 'OOD-both': 'orange'}
        
        # D scatter plots
        ax = axes[0, 0]
        for cat in ['ID', 'OOD-D', 'OOD-sigma', 'OOD-both']:
            if len(results[cat]['D_pred']) > 0:
                ax.scatter(results[cat]['D_true'], results[cat]['D_pred'], 
                          c=colors[cat], alpha=0.5, label=cat, s=20)
        
        # Add diagonal
        all_D = []
        for cat in results:
            all_D.extend(results[cat]['D_true'])
        if len(all_D) > 0:
            d_min, d_max = min(all_D), max(all_D)
            ax.plot([d_min, d_max], [d_min, d_max], 'k--', linewidth=2)
        
        ax.set_xlabel('True D (J/m²)')
        ax.set_ylabel('Predicted D (J/m²)')
        ax.set_title('D Predictions by Category')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # σ scatter plots
        ax = axes[0, 1]
        for cat in ['ID', 'OOD-D', 'OOD-sigma', 'OOD-both']:
            if len(results[cat]['sigma_pred']) > 0:
                ax.scatter(results[cat]['sigma_true'], results[cat]['sigma_pred'],
                          c=colors[cat], alpha=0.5, label=cat, s=20)
        
        all_s = []
        for cat in results:
            all_s.extend(results[cat]['sigma_true'])
        if len(all_s) > 0:
            s_min, s_max = min(all_s), max(all_s)
            ax.plot([s_min, s_max], [s_min, s_max], 'k--', linewidth=2)
        
        ax.set_xlabel('True σ')
        ax.set_ylabel('Predicted σ')
        ax.set_title('σ Predictions by Category')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # MAPE comparison bar chart
        ax = axes[0, 2]
        categories = [cat for cat in metrics if metrics[cat]['n_samples'] > 0]
        x = np.arange(len(categories))
        width = 0.35
        
        D_mapes = [metrics[cat]['D_MAPE'] for cat in categories]
        s_mapes = [metrics[cat]['sigma_MAPE'] for cat in categories]
        
        ax.bar(x - width/2, D_mapes, width, label='D MAPE', color='steelblue')
        ax.bar(x + width/2, s_mapes, width, label='σ MAPE', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('MAPE (%)')
        ax.set_title('MAPE by Category')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # D error distribution
        ax = axes[1, 0]
        for cat in ['ID', 'OOD-D']:
            if len(results[cat]['D_pred']) > 0:
                errors = np.array(results[cat]['D_pred']) - np.array(results[cat]['D_true'])
                ax.hist(errors * 1e3, bins=20, alpha=0.5, label=cat, color=colors[cat])
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('D Error (mJ/m²)')
        ax.set_ylabel('Count')
        ax.set_title('D Error Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # σ error distribution
        ax = axes[1, 1]
        for cat in ['ID', 'OOD-sigma']:
            if len(results[cat]['sigma_pred']) > 0:
                errors = np.array(results[cat]['sigma_pred']) - np.array(results[cat]['sigma_true'])
                ax.hist(errors, bins=20, alpha=0.5, label=cat, color=colors[cat])
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('σ Error')
        ax.set_ylabel('Count')
        ax.set_title('σ Error Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Summary metrics table
        ax = axes[1, 2]
        ax.axis('off')
        
        table_data = []
        headers = ['Category', 'N', 'D MAPE', 'σ MAPE', 'D R²', 'σ R²']
        
        for cat in ['ID', 'OOD-D', 'OOD-sigma', 'OOD-both']:
            if metrics[cat]['n_samples'] > 0:
                table_data.append([
                    cat,
                    f"{metrics[cat]['n_samples']}",
                    f"{metrics[cat]['D_MAPE']:.1f}%",
                    f"{metrics[cat]['sigma_MAPE']:.1f}%",
                    f"{metrics[cat]['D_R2']:.3f}",
                    f"{metrics[cat]['sigma_R2']:.3f}",
                ])
        
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                            loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        plt.suptitle('Out-of-Distribution Evaluation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# ==============================================================================
# GRADIENT ANALYSIS VISUALIZATION
# ==============================================================================

def plot_gradient_conflict_analysis(conflict_detector: GradientConflictDetector,
                                    save_path: Optional[Path] = None):
    """
    Visualize gradient conflicts during training.
    """
    history = conflict_detector.history
    
    if len(history['grad_norm_reg']) == 0:
        print("No gradient data to plot.")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    steps = np.arange(len(history['grad_norm_reg']))
    
    # Gradient magnitudes
    ax = axes[0, 0]
    ax.semilogy(steps, history['grad_norm_reg'], label='Regression', alpha=0.7)
    ax.semilogy(steps, history['grad_norm_energy'], label='Energy', alpha=0.7)
    ax.semilogy(steps, history['grad_norm_spectral'], label='Spectral', alpha=0.7)
    ax.set_xlabel('Step (×tracking_interval)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Magnitudes')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Cosine similarities
    ax = axes[0, 1]
    ax.plot(steps, history['cosine_reg_energy'], label='Reg ↔ Energy', alpha=0.7)
    ax.plot(steps, history['cosine_reg_spectral'], label='Reg ↔ Spectral', alpha=0.7)
    ax.plot(steps, history['cosine_energy_spectral'], label='Energy ↔ Spectral', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Gradient Alignment (>0: agree, <0: conflict)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    # Combined gradient norm
    ax = axes[0, 2]
    ax.semilogy(steps, history['grad_norm_combined'], color='purple', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Combined Gradient Norm')
    ax.set_title('Effective Gradient Magnitude')
    ax.grid(alpha=0.3)
    
    # Conflict rate over time (rolling window)
    ax = axes[1, 0]
    window = min(50, len(steps) // 5)
    if window > 1:
        conflicts_re = np.array(history['cosine_reg_energy']) < 0
        conflicts_rs = np.array(history['cosine_reg_spectral']) < 0
        
        # Rolling mean
        from scipy.ndimage import uniform_filter1d
        conflict_rate_re = uniform_filter1d(conflicts_re.astype(float), window)
        conflict_rate_rs = uniform_filter1d(conflicts_rs.astype(float), window)
        
        ax.plot(steps, conflict_rate_re * 100, label='Reg ↔ Energy conflicts', alpha=0.7)
        ax.plot(steps, conflict_rate_rs * 100, label='Reg ↔ Spectral conflicts', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Conflict Rate (%)')
        ax.set_title(f'Gradient Conflict Rate (rolling window={window})')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Distribution of cosine similarities
    ax = axes[1, 1]
    ax.hist(history['cosine_reg_energy'], bins=30, alpha=0.5, label='Reg ↔ Energy')
    ax.hist(history['cosine_reg_spectral'], bins=30, alpha=0.5, label='Reg ↔ Spectral')
    ax.axvline(0, color='k', linestyle='--', linewidth=2)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Gradient Alignments')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = conflict_detector.get_conflict_summary()
    
    text = "Gradient Conflict Summary\n" + "="*30 + "\n\n"
    text += f"Mean cosine (Reg ↔ Energy): {summary.get('mean_cosine_reg_energy', 0):.3f}\n"
    text += f"Mean cosine (Reg ↔ Spectral): {summary.get('mean_cosine_reg_spectral', 0):.3f}\n"
    text += f"Mean cosine (Energy ↔ Spectral): {summary.get('mean_cosine_energy_spectral', 0):.3f}\n\n"
    text += f"Conflict rate (Reg ↔ Energy): {summary.get('conflict_rate_reg_energy', 0)*100:.1f}%\n"
    text += f"Conflict rate (Reg ↔ Spectral): {summary.get('conflict_rate_reg_spectral', 0)*100:.1f}%\n"
    
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Gradient Conflict Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_improved_pinn(
    model: nn.Module,
    train_loader,
    val_loader,
    phys_config: PhysicsConfig,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    loss_weights: Dict[str, float],
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-4,
    use_annealing: bool = True,
    warmup_epochs: int = 20,
    track_gradients: bool = True,
    gradient_track_interval: int = 10,
    checkpoint_dir: Optional[Path] = None,
    run_tag: str = "improved_pinn"
) -> Dict:
    """
    Train the improved PINN with all enhancements.
    
    Returns:
        dict with training history, gradient analysis, and best model state
    """
    
    # Initialize components
    criterion = ImprovedPhysicsLoss(
        phys_config=phys_config,
        target_mean=target_mean,
        target_std=target_std,
        loss_weights=loss_weights
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Optional components
    annealing_schedule = None
    if use_annealing:
        annealing_schedule = LossAnnealingSchedule(
            final_weights=loss_weights,
            warmup_epochs=warmup_epochs,
            schedule='linear'
        )
    
    conflict_detector = None
    if track_gradients:
        conflict_detector = GradientConflictDetector()
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_reg': [], 'train_energy': [], 'train_spectral': [],
        'val_reg': [], 'val_energy': [], 'val_spectral': [],
        'lr': [],
        'val_mape_D': [], 'val_mape_sigma': [],
        'weights_energy': [], 'weights_spectral': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    t_mean_t = torch.tensor(target_mean, device=device, dtype=torch.float32)
    t_std_t = torch.tensor(target_std, device=device, dtype=torch.float32)
    
    batch_counter = 0
    
    print(f"Starting Improved PINN training for {epochs} epochs...")
    print(f"  Physics losses: energy_ratio={loss_weights.get('energy_ratio', 0)}, "
          f"spectral={loss_weights.get('spectral', 0)}")
    if use_annealing:
        print(f"  Annealing: warmup={warmup_epochs} epochs")
    print("-" * 80)
    
    for epoch in range(1, epochs + 1):
        
        # Get current weights
        if annealing_schedule:
            current_weights = annealing_schedule.get_weights(epoch, epochs)
            criterion.base_weights = current_weights
        else:
            current_weights = loss_weights
        
        # Record weights
        history['weights_energy'].append(current_weights.get('energy_ratio', 0))
        history['weights_spectral'].append(current_weights.get('spectral', 0))
        
        # ==================== TRAINING ====================
        model.train()
        running_loss = 0.0
        running_reg = 0.0
        running_energy = 0.0
        running_spectral = 0.0
        n_batches = 0
        
        for imgs, m_fields, targets, run_ids in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            if m_fields is not None:
                m_fields = {k: v.to(device) for k, v in m_fields.items()}
            
            optimizer.zero_grad()
            
            track_this_batch = track_gradients and (batch_counter % gradient_track_interval == 0)
            
            if track_this_batch:
                # Gradient tracking mode (no AMP)
                outputs = model(imgs)
                total_loss, loss_dict = criterion(
                    outputs, targets, m_fields, imgs, return_components=True
                )
                
                # Track gradient conflicts
                if conflict_detector is not None:
                    components = loss_dict['components']
                    conflict_detector.track_gradients(
                        model,
                        {'reg': components['loss_reg'],
                         'energy': components['loss_energy'],
                         'spectral': components['loss_spectral']},
                        current_weights
                    )
                
                # Standard backward
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                # Standard AMP training
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    total_loss, loss_dict = criterion(outputs, targets, m_fields, imgs)
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            running_loss += loss_dict['total']
            running_reg += loss_dict['regression']
            running_energy += loss_dict['energy']
            running_spectral += loss_dict['spectral']
            n_batches += 1
            batch_counter += 1
        
        epoch_train_loss = running_loss / n_batches
        epoch_train_reg = running_reg / n_batches
        epoch_train_energy = running_energy / n_batches
        epoch_train_spectral = running_spectral / n_batches
        
        history['train_loss'].append(epoch_train_loss)
        history['train_reg'].append(epoch_train_reg)
        history['train_energy'].append(epoch_train_energy)
        history['train_spectral'].append(epoch_train_spectral)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_reg = 0.0
        val_energy = 0.0
        val_spectral = 0.0
        n_val = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for imgs, m_fields, targets, run_ids in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                if m_fields is not None:
                    m_fields = {k: v.to(device) for k, v in m_fields.items()}
                
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    total_loss, loss_dict = criterion(outputs, targets, m_fields, imgs)
                
                val_loss += loss_dict['total']
                val_reg += loss_dict['regression']
                val_energy += loss_dict['energy']
                val_spectral += loss_dict['spectral']
                n_val += 1
                
                # Denormalize for metrics
                preds_unnorm = outputs * t_std_t + t_mean_t
                targets_unnorm = targets * t_std_t + t_mean_t
                all_preds.append(preds_unnorm.cpu())
                all_targets.append(targets_unnorm.cpu())
        
        epoch_val_loss = val_loss / n_val
        epoch_val_reg = val_reg / n_val
        epoch_val_energy = val_energy / n_val
        epoch_val_spectral = val_spectral / n_val
        
        history['val_loss'].append(epoch_val_loss)
        history['val_reg'].append(epoch_val_reg)
        history['val_energy'].append(epoch_val_energy)
        history['val_spectral'].append(epoch_val_spectral)
        
        # Compute MAPE
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        D_pred, D_true = all_preds[:, 0], all_targets[:, 0]
        s_pred, s_true = all_preds[:, 1], all_targets[:, 1]
        
        mape_D = torch.mean(torch.abs((D_pred - D_true) / (D_true + 1e-10))).item() * 100
        mape_s = torch.mean(torch.abs((s_pred - s_true) / (s_true + 1e-10))).item() * 100
        
        history['val_mape_D'].append(mape_D)
        history['val_mape_sigma'].append(mape_s)
        
        # Scheduler step
        scheduler.step(epoch_val_loss)
        lr_now = optimizer.param_groups[0]['lr']
        history['lr'].append(lr_now)
        
        # Checkpoint
        saved = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            saved = " *"
            
            if checkpoint_dir:
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch,
                    'val_loss': epoch_val_loss,
                    'history': history,
                    'gradient_history': conflict_detector.history if conflict_detector else None,
                    'target_mean': target_mean,
                    'target_std': target_std,
                    'phys_config': phys_config,
                    'loss_weights': loss_weights
                }, checkpoint_dir / f"best_{run_tag}.pt")
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1 or saved:
            print(f"Epoch {epoch:03d} | "
                  f"Train {epoch_train_loss:.3e} | Val {epoch_val_loss:.3e} | "
                  f"D MAPE {mape_D:.1f}% | σ MAPE {mape_s:.1f}% | "
                  f"LR {lr_now:.1e}{saved}")
    
    print("-" * 80)
    print(f"Training complete! Best val loss: {best_val_loss:.4e}")
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {
        'history': history,
        'best_model_state': best_model_state,
        'best_val_loss': best_val_loss,
        'conflict_detector': conflict_detector
    }


# ==============================================================================
# UTILITY: Print configuration summary
# ==============================================================================

def print_config_summary(phys_config: PhysicsConfig, loss_weights: Dict[str, float]):
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("IMPROVED PINN CONFIGURATION")
    print("="*60)
    print(f"\nPhysical Parameters:")
    print(f"  Exchange stiffness (Aex):  {phys_config.Aex:.2e} J/m")
    print(f"  Saturation mag (Ms):       {phys_config.Ms:.2e} A/m")
    print(f"  Anisotropy (Ku):           {phys_config.Ku:.2e} J/m³")
    print(f"  Cell size:                 {phys_config.cell_x*1e9:.1f} nm")
    print(f"\nDerived Quantities:")
    print(f"  Exchange length:           {phys_config.exchange_length*1e9:.2f} nm")
    print(f"  Quality factor Q:          {phys_config.quality_factor:.2e}")
    print(f"  Critical DMI (D_c):        {phys_config.critical_dmi*1e3:.3f} mJ/m²")
    print(f"  Domain wall width (δ):     {phys_config.domain_wall_width*1e9:.2f} nm")
    print(f"\nLoss Weights:")
    for key, val in loss_weights.items():
        print(f"  {key}: {val}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    config = PhysicsConfig()
    print_config_summary(config, {'regression': 1.0, 'energy_ratio': 0.3, 'spectral': 0.2})

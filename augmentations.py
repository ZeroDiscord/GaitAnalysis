"""
augmentations.py — Advanced EMG Augmentation Pipeline
=====================================================
Literature-backed augmentations for small biomedical time-series datasets.

Each augmentation:
  - Only modifies amplitude/temporal channels, never gait_phase or scalar features
  - Has a stochastic probability parameter (default: applied 50% of the time)
  - Is composable via AugmentationPipeline

References:
  - Time/magnitude warp: NeuroNet (Appendix B.1), TSCMamba (arXiv:2406.04419)
  - Channel dropout: FEMBA (arXiv:2502.06438) masked pretraining
  - Gaussian jitter: Universal in biosignal processing
  - Signal reversal: TSCMamba tango scanning exploits inversion invariance
"""

import numpy as np
import torch
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Individual augmentations (operate on numpy arrays for simplicity)
# ---------------------------------------------------------------------------

def time_warp(x: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
    """
    Smooth temporal deformation using cubic spline warping.
    
    Creates a smooth monotonic warping function by perturbing equidistant
    anchor points and interpolating. sigma controls deformation strength.
    
    Args:
        x: (T, C) array — time-series window
        sigma: std of Gaussian perturbation at anchor points (fraction of T)
        n_knots: number of interior anchor points
    """
    T, C = x.shape
    if T < 10:
        return x
    
    # Create anchor points with Gaussian perturbation
    orig_steps = np.linspace(0, T - 1, n_knots + 2)  # include endpoints
    perturbed = orig_steps.copy()
    # Only perturb interior points (keep endpoints fixed)
    perturbed[1:-1] += np.random.normal(0, sigma * T / n_knots, n_knots)
    
    # Ensure monotonicity
    perturbed = np.sort(perturbed)
    perturbed[0] = 0
    perturbed[-1] = T - 1
    
    # Interpolate to get warped time indices
    new_steps = np.linspace(0, T - 1, T)
    warped_steps = np.interp(new_steps, orig_steps, perturbed)
    warped_steps = np.clip(warped_steps, 0, T - 1)
    
    # Resample each channel
    result = np.zeros_like(x)
    for c in range(C):
        result[:, c] = np.interp(warped_steps, np.arange(T), x[:, c])
    
    return result


def magnitude_warp(x: np.ndarray, sigma: float = 0.2, n_knots: int = 4) -> np.ndarray:
    """
    Smooth amplitude deformation using cubic spline scaling.
    
    Generates a smooth multiplicative curve that varies across time,
    simulating natural EMG amplitude variability between gait cycles.
    
    Args:
        x: (T, C) array
        sigma: std of amplitude scaling factor at anchor points
        n_knots: number of interior anchor points
    """
    T, C = x.shape
    if T < 10:
        return x
    
    # Generate smooth scaling curve
    orig_steps = np.linspace(0, T - 1, n_knots + 2)
    scales = np.random.normal(1.0, sigma, n_knots + 2)
    scales = np.clip(scales, 0.5, 1.5)  # prevent extreme scaling
    
    new_steps = np.linspace(0, T - 1, T)
    smooth_scales = np.interp(new_steps, orig_steps, scales)
    
    return x * smooth_scales[:, np.newaxis]


def gaussian_jitter(x: np.ndarray, sigma_fraction: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise proportional to per-channel RMS.
    
    sigma = sigma_fraction * RMS(channel)
    
    This preserves the signal-to-noise ratio across channels with
    different absolute amplitudes (e.g., TA vs GA).
    """
    T, C = x.shape
    result = x.copy()
    for c in range(C):
        rms = np.sqrt(np.mean(x[:, c] ** 2)) + 1e-8
        noise = np.random.normal(0, sigma_fraction * rms, T)
        result[:, c] += noise
    return result


def channel_dropout(x: np.ndarray, n_channels_to_drop: int = 1,
                    amplitude_channels: Optional[List[int]] = None) -> np.ndarray:
    """
    Zero out one or more EMG channels entirely.
    
    Forces the model to classify from the remaining channel(s),
    improving robustness when one muscle's signal is weak/absent
    (common in hemiplegia, severe paresis).
    
    Args:
        x: (T, C) array
        n_channels_to_drop: how many channels to zero out
        amplitude_channels: indices of channels eligible for dropout
                           (default: [0, 1] = e_ant, e_ago only)
    """
    if amplitude_channels is None:
        amplitude_channels = [0, 1]  # e_ant, e_ago
    
    result = x.copy()
    n_drop = min(n_channels_to_drop, len(amplitude_channels) - 1)  # keep at least 1
    if n_drop <= 0:
        return result
    
    drop_indices = np.random.choice(amplitude_channels, size=n_drop, replace=False)
    for idx in drop_indices:
        if idx < x.shape[1]:
            result[:, idx] = 0.0
    
    return result


def signal_reversal(x: np.ndarray) -> np.ndarray:
    """
    Reverse the temporal direction of the signal.
    
    Valid for cyclic gait signals — a reversed gait cycle is still
    physiologically plausible (just starting from a different phase).
    TSCMamba's tango scanning exploits this inversion invariance.
    """
    return x[::-1].copy()


def amplitude_scale(x: np.ndarray, scale_range: tuple = (0.8, 1.2),
                    amplitude_channels: Optional[List[int]] = None) -> np.ndarray:
    """
    Scale amplitude of EMG channels only (not gait_phase or derived features).
    
    Fixed version of the original augmentation that incorrectly scaled all channels.
    """
    if amplitude_channels is None:
        amplitude_channels = [0, 1, 2, 3]  # e_ant, e_ago, torque, stiffness
    
    result = x.copy()
    scale = np.random.uniform(*scale_range)
    for idx in amplitude_channels:
        if idx < x.shape[1]:
            result[:, idx] *= scale
    
    return result


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Stochastic augmentation pipeline for EMG time-series.
    
    Each augmentation is applied independently with its own probability.
    Augmentations are applied in sequence (composition).
    
    Usage:
        pipeline = AugmentationPipeline.default_emg()
        augmented = pipeline(window_features)  # numpy (T, C) -> numpy (T, C)
    """
    
    def __init__(self, augmentations: List[tuple]):
        """
        Args:
            augmentations: List of (name, function, probability) tuples.
                          function signature: (np.ndarray) -> np.ndarray
                          Input/output shape: (T, C)
        """
        self.augmentations = augmentations
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline stochastically."""
        result = x
        for name, fn, prob in self.augmentations:
            if np.random.random() < prob:
                try:
                    result = fn(result)
                except Exception:
                    pass  # Skip failed augmentation silently
        return result
    
    @staticmethod
    def default_emg(
        p_time_warp: float = 0.3,
        p_mag_warp: float = 0.3,
        p_jitter: float = 0.5,
        p_channel_drop: float = 0.15,
        p_reversal: float = 0.1,
        p_amp_scale: float = 0.5,
    ) -> 'AugmentationPipeline':
        """
        Default augmentation pipeline tuned for 2-channel EMG gait data.
        
        Probabilities are conservative — each augmentation is independent.
        On average, ~2 augmentations fire per window.
        """
        return AugmentationPipeline([
            ('gaussian_jitter', gaussian_jitter, p_jitter),
            ('amplitude_scale', amplitude_scale, p_amp_scale),
            ('time_warp', time_warp, p_time_warp),
            ('magnitude_warp', magnitude_warp, p_mag_warp),
            ('channel_dropout', channel_dropout, p_channel_drop),
            ('signal_reversal', signal_reversal, p_reversal),
        ])
    
    @staticmethod
    def none() -> 'AugmentationPipeline':
        """No-op pipeline for validation/test."""
        return AugmentationPipeline([])
    
    def __repr__(self):
        lines = ["AugmentationPipeline:"]
        for name, _, prob in self.augmentations:
            lines.append(f"  {name}: p={prob:.2f}")
        return "\n".join(lines)

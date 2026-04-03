"""Data augmentation for sEMG windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfilt


def gaussian_noise(window: np.ndarray, rng: np.random.Generator, std_frac: float = 0.05) -> np.ndarray:
    """Add Gaussian noise scaled to per-channel RMS."""
    rms = np.sqrt(np.mean(window ** 2, axis=0, keepdims=True)) + 1e-8
    return window + rng.normal(0, std_frac * rms, window.shape)


def amplitude_scale(window: np.ndarray, rng: np.random.Generator, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    """Scale amplitude per channel independently."""
    scales = rng.uniform(low, high, (1, window.shape[1]))
    return window * scales


def time_warp(window: np.ndarray, rng: np.random.Generator, sigma: float = 0.2) -> np.ndarray:
    """Random time warping via cubic spline interpolation."""
    n, c = window.shape
    tt = np.arange(n, dtype=float)
    knots = np.sort(rng.uniform(0, n, 4))
    knots = np.clip(knots, 0, n - 1)
    warped = np.zeros_like(window)
    cs = CubicSpline([0, *knots, n - 1], [0, *(knots + rng.normal(0, sigma * n, 4)), n - 1])
    new_t = np.clip(cs(tt), 0, n - 1)
    for ch in range(c):
        warped[:, ch] = np.interp(new_t, tt, window[:, ch])
    return warped.astype(window.dtype)


def channel_dropout(window: np.ndarray, rng: np.random.Generator, max_drop: int = 2) -> np.ndarray:
    """Zero out up to max_drop random channels."""
    n_drop = rng.integers(0, max_drop + 1)
    if n_drop == 0:
        return window
    out = window.copy()
    drop_idx = rng.choice(window.shape[1], n_drop, replace=False)
    out[:, drop_idx] = 0.0
    return out


def time_shift(window: np.ndarray, rng: np.random.Generator, max_shift_frac: float = 0.1) -> np.ndarray:
    """Circular time shift."""
    max_shift = max(1, int(max_shift_frac * window.shape[0]))
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(window, shift, axis=0)


def band_stop_noise(window: np.ndarray, rng: np.random.Generator, fs: float = 200.0) -> np.ndarray:
    """Add 60Hz band-stop shaped noise (simulates power line interference)."""
    noise = rng.normal(0, 0.01, window.shape)
    freq = 60.0 / (fs / 2)
    freq = np.clip(freq, 0.01, 0.99)
    b_sos = butter(2, [max(0.01, freq - 0.05), min(0.99, freq + 0.05)], btype="band", output="sos")
    noise = sosfilt(b_sos, noise, axis=0)
    return (window + noise).astype(window.dtype)


def electrode_offset(window: np.ndarray, rng: np.random.Generator, max_offset: float = 0.05) -> np.ndarray:
    """Add a constant DC offset per channel (electrode drift simulation)."""
    offsets = rng.uniform(-max_offset, max_offset, (1, window.shape[1]))
    return window + offsets


def magnitude_warp(window: np.ndarray, rng: np.random.Generator, sigma: float = 0.1) -> np.ndarray:
    """Smooth amplitude modulation via cubic spline."""
    n, c = window.shape
    knots_x = np.linspace(0, n - 1, 4)
    warped = np.zeros_like(window)
    for ch in range(c):
        knots_y = 1.0 + rng.normal(0, sigma, 4)
        cs = CubicSpline(knots_x, knots_y)
        scale = np.clip(cs(np.arange(n)), 0.2, 2.0)
        warped[:, ch] = window[:, ch] * scale
    return warped.astype(window.dtype)


@dataclass
class AugmentationPipeline:
    """Configurable pipeline of augmentation functions."""
    augmentations: List = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        out = window.copy()
        for aug, prob in zip(self.augmentations, self.probabilities):
            if self.rng.random() < prob:
                out = aug(out, self.rng)
        return out


def get_default_pipeline(strength: str = "medium", seed: Optional[int] = None) -> AugmentationPipeline:
    """Create a pre-configured augmentation pipeline."""
    all_augs = [
        gaussian_noise, amplitude_scale, time_warp,
        channel_dropout, time_shift, band_stop_noise,
        electrode_offset, magnitude_warp,
    ]
    probs = {
        "light":  [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "medium": [0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.2, 0.3],
        "strong": [0.8, 0.7, 0.5, 0.4, 0.5, 0.4, 0.4, 0.5],
    }
    return AugmentationPipeline(
        augmentations=all_augs,
        probabilities=probs[strength],
        seed=seed,
    )


def mixup_emg(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    MixUp augmentation for EMG windows.

    Args:
        X: (N, window_samples, N_channels)
        y: (N,) integer class labels
        alpha: Beta distribution parameter
    Returns:
        X_mixed, y_a, y_b, lam  (for mixed loss)
    """
    if rng is None:
        rng = np.random.default_rng()
    lam = rng.beta(alpha, alpha)
    n = X.shape[0]
    idx = rng.permutation(n)
    X_mixed = lam * X + (1 - lam) * X[idx]
    return X_mixed, y, y[idx], lam

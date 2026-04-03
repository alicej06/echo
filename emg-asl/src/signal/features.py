"""Feature extraction from sEMG windows."""
from __future__ import annotations

import numpy as np


def _rms(window: np.ndarray) -> np.ndarray:
    """Root Mean Square per channel. window: (samples, channels)"""
    return np.sqrt(np.mean(window ** 2, axis=0))


def _mav(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value per channel."""
    return np.mean(np.abs(window), axis=0)


def _wl(window: np.ndarray) -> np.ndarray:
    """Waveform Length per channel."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def _zc(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Zero Crossing count per channel."""
    diffs = np.diff(np.sign(window), axis=0)
    crossings = (np.abs(diffs) >= 2 * threshold).astype(float)
    return np.sum(crossings, axis=0)


def _ssc(window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Slope Sign Change count per channel."""
    diff1 = np.diff(window, axis=0)
    diff2 = np.diff(diff1, axis=0)
    sign_change = np.sign(diff1[:-1]) != np.sign(diff1[1:])
    magnitude = np.abs(diff2) >= threshold
    return np.sum(sign_change & magnitude, axis=0).astype(float)


def _var(window: np.ndarray) -> np.ndarray:
    """Variance per channel."""
    return np.var(window, axis=0)


def _ar4(window: np.ndarray) -> np.ndarray:
    """AR(4) coefficients (first 4 auto-regression) - concatenated."""
    from numpy.linalg import lstsq
    n, c = window.shape
    order = 4
    results = np.zeros((order, c))
    for ch in range(c):
        x = window[:, ch]
        X = np.column_stack([x[order - k - 1: n - k - 1] for k in range(order)])
        y = x[order:]
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        results[:, ch] = coeffs
    # Average over 4 AR coefficients → 1 value per channel
    return results.mean(axis=0)


def _iemg(window: np.ndarray) -> np.ndarray:
    """Integrated EMG (sum of absolute values)."""
    return np.sum(np.abs(window), axis=0)


def _kurt(window: np.ndarray) -> np.ndarray:
    """Kurtosis per channel."""
    from scipy.stats import kurtosis
    return kurtosis(window, axis=0)


def _mnf(window: np.ndarray, fs: float = 2000.0) -> np.ndarray:
    """Mean frequency per channel."""
    freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / fs)
    psd = np.abs(np.fft.rfft(window, axis=0)) ** 2
    total_power = psd.sum(axis=0)
    mean_freq = np.dot(freqs, psd) / (total_power + 1e-8)
    return mean_freq


def extract_features(window: np.ndarray, fs: float = 2000.0) -> np.ndarray:
    """
    Extract 10 features per channel from a (samples, channels) window.
    Returns 1-D vector of length N_CHANNELS * 10 = 160.
    """
    features = np.stack([
        _rms(window),
        _mav(window),
        _wl(window),
        _zc(window),
        _ssc(window),
        _var(window),
        _ar4(window),
        _iemg(window),
        _kurt(window),
        _mnf(window, fs=fs),
    ], axis=0)  # (10, N_CHANNELS)
    return features.T.flatten()  # (N_CHANNELS * 10,) = (80,)


def window_signal(
    signal: np.ndarray,
    window_samples: int = 400,
    hop_samples: int = 200,
) -> np.ndarray:
    """
    Slide a window over signal and extract feature vectors.

    Args:
        signal: (total_samples, N_CHANNELS)
        window_samples: samples per window (default 400 = 200ms @ 2kHz)
        hop_samples: hop size (default 200 = 50% overlap)

    Returns:
        features: (n_windows, feature_dim)
    """
    n_samples, n_channels = signal.shape
    starts = range(0, n_samples - window_samples + 1, hop_samples)
    windows = [signal[s: s + window_samples] for s in starts]
    return np.array([extract_features(w) for w in windows])

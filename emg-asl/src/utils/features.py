"""
EMG feature extraction — time-domain and frequency-domain features.

All functions accept a window of shape ``(n_samples, n_channels)`` and
return either a 2-D array of shape ``(n_features_per_channel, n_channels)``
or a flat 1-D feature vector ready for the classifier.

Time-domain features (5 per channel)
--------------------------------------
RMS   — Root Mean Square amplitude
MAV   — Mean Absolute Value
WL    — Waveform Length (cumulative arc length)
ZC    — Zero Crossing rate
SSC   — Slope Sign Change rate

Frequency-domain features (5 per channel)
-------------------------------------------
mean_freq    — power-weighted mean frequency
median_freq  — frequency at which the PSD is split 50/50 by cumulative power
moment2      — 2nd spectral moment (variance around mean_freq)
moment3      — 3rd spectral moment (skewness of PSD)
moment4      — 4th spectral moment (kurtosis of PSD)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_window(window: np.ndarray) -> None:
    if window.ndim != 2:
        raise ValueError(
            f"window must be 2-D (n_samples, n_channels), got shape {window.shape}"
        )


# ---------------------------------------------------------------------------
# Time-domain feature extraction
# ---------------------------------------------------------------------------


def extract_time_features(window: np.ndarray) -> np.ndarray:
    """Compute 5 time-domain features for every channel.

    Parameters
    ----------
    window:
        EMG window, shape ``(n_samples, n_channels)``, float.

    Returns
    -------
    np.ndarray
        Feature matrix, shape ``(5, n_channels)``.
        Rows: [RMS, MAV, WL, ZC, SSC].
    """
    _check_window(window)
    n_samples, n_channels = window.shape

    # --- RMS: root-mean-square amplitude
    rms = np.sqrt(np.mean(window ** 2, axis=0))  # (n_channels,)

    # --- MAV: mean absolute value
    mav = np.mean(np.abs(window), axis=0)

    # --- WL: waveform length — sum of absolute sample-to-sample differences
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

    # --- ZC: zero crossing count (normalised by n_samples)
    # A crossing occurs when consecutive samples have opposite signs.
    signs = np.sign(window)
    # Replace zeros with the sign of the previous non-zero sample to avoid
    # counting flat regions as crossings; fall back to +1 if all zeros.
    for ch in range(n_channels):
        prev_sign = 1
        for i in range(n_samples):
            if signs[i, ch] == 0:
                signs[i, ch] = prev_sign
            else:
                prev_sign = signs[i, ch]
    zc_raw = np.sum(np.abs(np.diff(signs, axis=0)) > 0, axis=0).astype(float)
    zc = zc_raw / max(n_samples - 1, 1)

    # --- SSC: slope sign change count (normalised by n_samples)
    # A slope sign change occurs when the difference between consecutive samples
    # changes sign (local maxima / minima in the signal).
    diffs = np.diff(window, axis=0)          # (n_samples-1, n_channels)
    diff_signs = np.sign(diffs)
    ssc_raw = np.sum(np.abs(np.diff(diff_signs, axis=0)) > 0, axis=0).astype(float)
    ssc = ssc_raw / max(n_samples - 2, 1)

    return np.stack([rms, mav, wl, zc, ssc], axis=0)  # (5, n_channels)


# ---------------------------------------------------------------------------
# Frequency-domain feature extraction
# ---------------------------------------------------------------------------


def extract_freq_features(window: np.ndarray, fs: float) -> np.ndarray:
    """Compute 5 frequency-domain features for every channel using Welch PSD.

    Parameters
    ----------
    window:
        EMG window, shape ``(n_samples, n_channels)``, float.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Feature matrix, shape ``(5, n_channels)``.
        Rows: [mean_freq, median_freq, moment2, moment3, moment4].
    """
    _check_window(window)
    n_samples, n_channels = window.shape

    # Use a segment length of at most the window size; nperseg must be ≥ 1.
    nperseg = min(n_samples, max(4, n_samples // 2))

    mean_freqs = np.empty(n_channels)
    median_freqs = np.empty(n_channels)
    moments2 = np.empty(n_channels)
    moments3 = np.empty(n_channels)
    moments4 = np.empty(n_channels)

    for ch in range(n_channels):
        freqs, psd = welch(
            window[:, ch].astype(np.float64),
            fs=fs,
            nperseg=nperseg,
            scaling="density",
        )
        psd_sum = psd.sum() + 1e-12  # avoid div-by-zero on silent channels

        # Normalised PSD (probability distribution over frequency)
        p = psd / psd_sum

        # Weighted mean frequency
        mf = np.dot(freqs, p)
        mean_freqs[ch] = mf

        # Median frequency — smallest f_k such that cumulative power >= 50%
        cumulative = np.cumsum(psd)
        half_power = cumulative[-1] / 2.0
        idx_med = np.searchsorted(cumulative, half_power)
        idx_med = min(idx_med, len(freqs) - 1)
        median_freqs[ch] = freqs[idx_med]

        # Spectral moments about the mean frequency
        delta = freqs - mf
        moments2[ch] = np.dot(delta ** 2, p)
        moments3[ch] = np.dot(delta ** 3, p)
        moments4[ch] = np.dot(delta ** 4, p)

    return np.stack(
        [mean_freqs, median_freqs, moments2, moments3, moments4], axis=0
    )  # (5, n_channels)


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------


def extract_features(window: np.ndarray, fs: float = 200.0) -> np.ndarray:
    """Extract and concatenate all time- and frequency-domain features.

    Parameters
    ----------
    window:
        EMG window, shape ``(n_samples, n_channels)``, float.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Flat 1-D feature vector of length ``10 * n_channels``.
        Order: [time_features (5*n_ch) | freq_features (5*n_ch)],
        where each block is channel-major (ch0_f0, ch0_f1, ..., ch1_f0, ...).
    """
    _check_window(window)
    time_feats = extract_time_features(window)     # (5, n_channels)
    freq_feats = extract_freq_features(window, fs)  # (5, n_channels)

    # Stack along feature axis -> (10, n_channels), then flatten column-major
    # so that all features for ch0 come first, then ch1, etc.
    combined = np.vstack([time_feats, freq_feats])  # (10, n_channels)
    return combined.T.flatten()                      # (10 * n_channels,)


# ---------------------------------------------------------------------------
# Feature name generation
# ---------------------------------------------------------------------------

_TIME_FEATURE_NAMES: list[str] = ["RMS", "MAV", "WL", "ZC", "SSC"]
_FREQ_FEATURE_NAMES: list[str] = [
    "mean_freq",
    "median_freq",
    "spectral_moment2",
    "spectral_moment3",
    "spectral_moment4",
]
_ALL_FEATURE_NAMES: list[str] = _TIME_FEATURE_NAMES + _FREQ_FEATURE_NAMES


def get_feature_names(n_channels: int) -> list[str]:
    """Return a list of feature name strings matching the output of ``extract_features``.

    Parameters
    ----------
    n_channels:
        Number of EMG channels.

    Returns
    -------
    list[str]
        Names like ``"ch0_RMS"``, ``"ch0_MAV"``, ..., ``"ch7_spectral_moment4"``.
        Length is ``10 * n_channels``.
    """
    names: list[str] = []
    for ch in range(n_channels):
        for feat in _ALL_FEATURE_NAMES:
            names.append(f"ch{ch}_{feat}")
    return names

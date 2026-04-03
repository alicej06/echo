"""
EMG signal filtering utilities.

All functions operate on numpy arrays of shape (n_samples, n_channels)
and return arrays of the same shape.  Filtering is applied independently
to each channel (column).

Filters
-------
bandpass_filter  — 4th-order Butterworth bandpass (default 20-450 Hz)
notch_filter     — IIR notch for power-line rejection (default 60 Hz)
dc_remove        — subtract per-channel mean (high-pass at DC)
normalize_signal — RMS normalisation per channel (unit RMS output)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_2d(signal: np.ndarray) -> None:
    """Raise ValueError when the array is not 2-D."""
    if signal.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array (n_samples, n_channels), got shape {signal.shape}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to each channel.

    Parameters
    ----------
    signal:
        Raw EMG array, shape ``(n_samples, n_channels)``, dtype float.
    lowcut:
        Lower corner frequency in Hz.
    highcut:
        Upper corner frequency in Hz.
    fs:
        Sampling frequency in Hz.
    order:
        Filter order.  The effective order is doubled by ``sosfiltfilt``
        (forward + backward pass).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape and dtype as *signal*.
    """
    _assert_2d(signal)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0.0 or high >= 1.0:
        raise ValueError(
            f"Normalised frequencies must be in (0, 1). Got low={low:.4f}, high={high:.4f}. "
            f"Check lowcut/highcut against fs={fs}."
        )

    sos = butter(order, [low, high], btype="band", output="sos")
    # sosfiltfilt applies the filter forward and backward for zero phase shift.
    out = np.empty_like(signal, dtype=np.float64)
    for ch in range(signal.shape[1]):
        out[:, ch] = sosfiltfilt(sos, signal[:, ch].astype(np.float64))
    return out


def notch_filter(
    signal: np.ndarray,
    freq: float,
    fs: float,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply a zero-phase IIR notch filter to suppress a single frequency.

    Parameters
    ----------
    signal:
        EMG array, shape ``(n_samples, n_channels)``.
    freq:
        Frequency to notch out, in Hz (e.g., 60 for US power-line noise).
    fs:
        Sampling frequency in Hz.
    quality:
        Quality factor ``Q = freq / bandwidth``.  Higher values produce a
        narrower notch.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as *signal*.
    """
    _assert_2d(signal)
    w0 = freq / (0.5 * fs)  # normalised frequency [0, 1]
    if not (0.0 < w0 < 1.0):
        raise ValueError(
            f"Notch frequency {freq} Hz is out of range for fs={fs} Hz."
        )
    b, a = iirnotch(w0, quality)
    out = np.empty_like(signal, dtype=np.float64)
    for ch in range(signal.shape[1]):
        out[:, ch] = filtfilt(b, a, signal[:, ch].astype(np.float64))
    return out


def dc_remove(signal: np.ndarray) -> np.ndarray:
    """Remove the DC offset from each channel by subtracting its mean.

    Parameters
    ----------
    signal:
        EMG array, shape ``(n_samples, n_channels)``.

    Returns
    -------
    np.ndarray
        Zero-mean signal, same shape and dtype as *signal*.
    """
    _assert_2d(signal)
    # Subtract per-channel mean along the sample axis (axis=0).
    return (signal - signal.mean(axis=0)).astype(np.float64)


def normalize_signal(signal: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """RMS-normalise each channel to unit RMS amplitude.

    Parameters
    ----------
    signal:
        EMG array, shape ``(n_samples, n_channels)``.
    eps:
        Small value added to the denominator to prevent division-by-zero
        on silent / flat channels.

    Returns
    -------
    np.ndarray
        Normalised signal; each channel has RMS ≈ 1.0.
    """
    _assert_2d(signal)
    sig = signal.astype(np.float64)
    rms = np.sqrt(np.mean(sig ** 2, axis=0))  # shape (n_channels,)
    return sig / (rms[np.newaxis, :] + eps)


def apply_full_filter_chain(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    notch_freq: float = 60.0,
    notch_quality: float = 30.0,
    bandpass_order: int = 4,
) -> np.ndarray:
    """Convenience wrapper: DC removal -> bandpass -> notch -> normalise.

    Parameters
    ----------
    signal:
        Raw EMG array, shape ``(n_samples, n_channels)``.
    fs:
        Sampling frequency in Hz.
    lowcut, highcut:
        Bandpass corner frequencies in Hz.
    notch_freq:
        Power-line frequency to reject in Hz.
    notch_quality:
        Notch quality factor.
    bandpass_order:
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Fully processed signal ready for feature extraction.
    """
    sig = dc_remove(signal)
    sig = bandpass_filter(sig, lowcut, highcut, fs, order=bandpass_order)
    sig = notch_filter(sig, notch_freq, fs, quality=notch_quality)
    sig = normalize_signal(sig)
    return sig

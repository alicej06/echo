"""Bandpass, notch, and rectification filters for sEMG signals."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    fs: float = 2000.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass filter. signal shape: (samples, channels)."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def notch_filter(
    signal: np.ndarray,
    freq: float = 60.0,
    fs: float = 2000.0,
    quality: float = 30.0,
) -> np.ndarray:
    """IIR notch filter to remove power-line interference."""
    b, a = iirnotch(freq / (fs / 2), quality)
    from scipy.signal import lfilter
    return lfilter(b, a, signal, axis=0)


def full_wave_rectify(signal: np.ndarray) -> np.ndarray:
    """Full-wave rectification (absolute value)."""
    return np.abs(signal)


def preprocess(signal: np.ndarray, fs: float = 2000.0) -> np.ndarray:
    """Full preprocessing pipeline: bandpass → notch → rectify."""
    sig = bandpass_filter(signal, fs=fs)
    sig = notch_filter(sig, fs=fs)
    sig = full_wave_rectify(sig)
    return sig

"""Dataset loading utilities for EMG-ASL data."""
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.constants import (
    SAMPLE_RATE, N_CHANNELS, WINDOW_SAMPLES, HOP_SAMPLES,
    N_CLASSES, ASL_CLASSES,
)
from src.signal.features import window_signal
from src.signal.filters import preprocess


def load_session(
    session_path: str,
    apply_preprocessing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single recording session from an .npz file.

    Expected keys in npz:
      - 'emg':    (total_samples, N_CHANNELS) float32
      - 'labels': (total_samples,) int32  (class index per sample)

    Returns:
        emg:    (total_samples, N_CHANNELS)
        labels: (total_samples,)
    """
    data = np.load(session_path)
    emg = data["emg"].astype(np.float32)
    labels = data["labels"].astype(np.int32)

    if apply_preprocessing:
        emg = preprocess(emg, fs=SAMPLE_RATE).astype(np.float32)

    return emg, labels


def create_windows(
    emg: np.ndarray,
    labels: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
    extract_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide window over EMG and assign majority-vote label.

    Args:
        emg:    (total_samples, N_CHANNELS)
        labels: (total_samples,)
        extract_features: if True returns feature vectors; else raw windows

    Returns:
        X: (n_windows, window_samples, N_CHANNELS) or (n_windows, feature_dim)
        y: (n_windows,)
    """
    n_samples = emg.shape[0]
    starts = list(range(0, n_samples - window_samples + 1, hop_samples))
    X_list, y_list = [], []

    for s in starts:
        win_emg = emg[s: s + window_samples]
        win_labels = labels[s: s + window_samples]
        # Majority vote
        counts = np.bincount(win_labels, minlength=N_CLASSES)
        label = int(np.argmax(counts))

        if extract_features:
            from src.signal.features import extract_features as ef
            X_list.append(ef(win_emg))
        else:
            X_list.append(win_emg)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def generate_synthetic_dataset(
    n_samples_per_class: int = 100,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic EMG dataset for development/testing.

    Each class has a distinct mean amplitude pattern across channels.

    Returns:
        X: (n_total, WINDOW_SAMPLES, N_CHANNELS)  float32
        y: (n_total,)  int64
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for class_idx in range(N_CLASSES):
        # Unique activation pattern per class
        pattern = np.zeros(N_CHANNELS)
        active = rng.choice(N_CHANNELS, size=rng.integers(2, 6), replace=False)
        pattern[active] = rng.uniform(0.3, 1.0, size=len(active))

        for _ in range(n_samples_per_class):
            # Sinusoidal carrier + pattern + noise
            t = np.linspace(0, 0.2, WINDOW_SAMPLES)
            freq = 20 + class_idx * 3  # Hz
            carrier = np.sin(2 * np.pi * freq * t)[:, None]  # (40, 1)
            window = carrier * pattern[None, :] + rng.normal(0, noise_std, (WINDOW_SAMPLES, N_CHANNELS))
            X_list.append(window.astype(np.float32))
            y_list.append(class_idx)

    X = np.array(X_list)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_dataset(
    data_dir: str,
    split: str = "train",
    extract_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all session .npz files from data_dir/split/.

    Returns combined (X, y) arrays.
    """
    pattern = os.path.join(data_dir, split, "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .npz files found at {pattern}")

    X_all, y_all = [], []
    for f in files:
        emg, labels = load_session(f)
        X, y = create_windows(emg, labels, extract_features=extract_features)
        X_all.append(X)
        y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)

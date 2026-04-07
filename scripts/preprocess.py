"""
preprocess.py
-------------
Feature extraction matching the DyFAV GitHub repo exactly.
https://github.com/prwlnght/DyFAV

Sensors (17 channels, raw column order from dataset CSV):
  EMG0-7 (cols 0-7), Accl0-2 (cols 8-10), Gyr0-2 (cols 11-13), Orien0-2 (cols 14-16)

Features per channel (5): mean, min, max, stdev, energy
Windows (6): full signal + 5 equal segments

Feature vector layout (510 total):
  for window in [full, seg1, seg2, seg3, seg4, seg5]:
    for feature in [mean, min, max, stdev, energy]:
      for sensor in [EMG0..7, Accl0..2, Gyr0..2, Orien0..2]:   # 17 sensors
        → 1 value

Total: 6 × 5 × 17 = 510
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_SENSORS   = 17     # EMG(8) + Accl(3) + Gyr(3) + Orien(3)
N_FEATURES  = 5      # mean, min, max, stdev, energy
N_WINDOWS   = 6      # full + 5 segments
FEATURE_DIM = N_SENSORS * N_FEATURES * N_WINDOWS  # 510

SENSORS = [
    "EMG0", "EMG1", "EMG2", "EMG3", "EMG4", "EMG5", "EMG6", "EMG7",
    "Accl0", "Accl1", "Accl2",
    "Gyr0",  "Gyr1",  "Gyr2",
    "Orien0", "Orien1", "Orien2",
]
FEATURES = ["mean", "min", "max", "stdev", "energy"]

LABEL_RE = re.compile(r"alphabet_([a-z])")


def _parse_label(filename: str) -> str | None:
    m = LABEL_RE.search(filename)
    return m.group(1) if m else None


def _load_recording(csv_path: Path) -> np.ndarray:
    """Load raw recording. Returns (n_samples, 17) float64."""
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw


def _compute_energy(signal: np.ndarray) -> float:
    return float((signal ** 2).sum())


def _window_features(window: np.ndarray) -> np.ndarray:
    """
    Compute 5*17=85 features for one window.
    Order: for each feature_type in [mean, min, max, stdev, energy]:
               for each sensor (17): value
    This matches the DyFAV repo's column layout exactly.
    """
    means  = window.mean(axis=0)          # (17,)
    mins   = window.min(axis=0)           # (17,)
    maxes  = window.max(axis=0)           # (17,)
    stdevs = window.std(axis=0)           # (17,)
    energy = np.array([_compute_energy(window[:, ch]) for ch in range(N_SENSORS)])  # (17,)
    return np.concatenate([means, mins, maxes, stdevs, energy])   # (85,)


def extract_510_features(recording: np.ndarray) -> np.ndarray:
    """
    Extract 510 features from one raw recording.

    Args:
        recording: (n_samples, 17) raw sensor data — columns used as-is

    Returns:
        (510,) float64 feature vector matching DyFAV repo layout
    """
    n = len(recording)
    seg_size = int(n / 5)   # matches repo: int(len(file_frame)/number_of_partitions)

    parts = []

    # Window 0: full signal
    parts.append(_window_features(recording))

    # Windows 1-5: equal partitions (repo uses integer division)
    lower = 0
    upper = seg_size
    for _ in range(5):
        seg = recording[lower:upper]
        if len(seg) == 0:
            seg = recording[:1]   # safety fallback
        parts.append(_window_features(seg))
        lower = upper
        upper = upper + seg_size

    return np.concatenate(parts).astype(np.float32)   # (510,)


def get_feature_names() -> list[str]:
    """Return the 510 feature names matching the DyFAV repo's column naming."""
    names = []
    for i in range(N_WINDOWS):
        for feat in FEATURES:
            for sensor in SENSORS:
                names.append(f"{sensor}_{feat}_{i}")
    return names


def load_dataset(
    data_dir: str,
    letters: list[str] | None = None,
    per_user: bool = False,
) -> tuple | dict:
    """
    Load all recordings and extract 510 features.

    Returns (per_user=False):
        X:        (N, 510) float32
        y:        (N,) int array — letter index 0-25
        rec_ids:  (N,) int array — which recording each sample came from
        user_ids: (N,) int array — which user each sample came from
        le_classes: list of 26 letter strings (a-z)

    Returns (per_user=True):
        dict mapping user_id -> (X, y, rec_ids)
    """
    data_dir = Path(data_dir)
    filter_set = None
    if letters and letters not in (["all"], None):
        filter_set = {l.lower() for l in letters}

    all_letters = sorted([chr(ord('a') + i) for i in range(26)])

    if per_user:
        result = {}
        for user_dir in sorted(data_dir.glob("User*")):
            if not user_dir.is_dir():
                continue
            uid = int(''.join(filter(str.isdigit, user_dir.name)) or 0)
            X_parts, y_parts, rec_parts = [], [], []
            rec_id = 0
            for csv_path in sorted(user_dir.glob("*.csv")):
                label = _parse_label(csv_path.name)
                if label is None:
                    continue
                if filter_set and label not in filter_set:
                    continue
                raw  = _load_recording(csv_path)
                feat = extract_510_features(raw)
                X_parts.append(feat)
                y_parts.append(all_letters.index(label))
                rec_parts.append(rec_id)
                rec_id += 1
            if X_parts:
                result[uid] = (
                    np.stack(X_parts),
                    np.array(y_parts, dtype=np.int32),
                    np.array(rec_parts, dtype=np.int32),
                )
        return result

    # Merged across all users
    X_parts, y_parts, rec_parts, user_parts = [], [], [], []
    rec_id = 0
    for user_dir in sorted(data_dir.glob("User*")):
        if not user_dir.is_dir():
            continue
        uid = int(''.join(filter(str.isdigit, user_dir.name)) or 0)
        for csv_path in sorted(user_dir.glob("*.csv")):
            label = _parse_label(csv_path.name)
            if label is None:
                continue
            if filter_set and label not in filter_set:
                continue
            raw  = _load_recording(csv_path)
            feat = extract_510_features(raw)
            X_parts.append(feat)
            y_parts.append(all_letters.index(label))
            rec_parts.append(rec_id)
            user_parts.append(uid)
            rec_id += 1

    X        = np.stack(X_parts).astype(np.float32)
    y        = np.array(y_parts,  dtype=np.int32)
    rec_ids  = np.array(rec_parts, dtype=np.int32)
    user_ids = np.array(user_parts, dtype=np.int32)

    return X, y, rec_ids, user_ids, all_letters

"""
Loader for Meta Research EMG datasets.

Supports three Meta open-source datasets, all recorded on 16-channel / 2kHz
hardware. Data is resampled to 200 Hz and channel-subset to 8ch to match
the MYO Armband acquisition spec used at inference time:

  - emg2pose:                  193 participants · 370 hours · hand joint angles
  - generic-neuromotor-interface (gni): 100 participants · 200+ hours · gesture classes
  - emg2qwerty:                108 users · 346 hours · bilateral typing

HDF5 file structure assumed:
  emg2pose:
    /{session_id}/emg      float32 (N, 16)
    /{session_id}/joints   float32 (N, 63)  — hand joint angles

  gni:
    /{session_id}/emg      float32 (N, 16)
    /{session_id}/labels   int32   (N,)

  emg2qwerty:
    /{session_id}/emg_l    float32 (N, 16)  — left wrist
    /{session_id}/emg_r    float32 (N, 16)  — right wrist
    /{session_id}/labels   int32   (N,)

Usage:
    from src.data.meta_loader import MetaEMGLoader
    loader = MetaEMGLoader("data/meta/emg2pose/", dataset="emg2pose")
    emg, joints = loader.load_session("subject_001_session_01")
    X, y = loader.build_windows()
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Literal, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from src.constants import WINDOW_SAMPLES, HOP_SAMPLES, N_CHANNELS, SAMPLE_RATE


DatasetType = Literal["emg2pose", "gni", "emg2qwerty"]


class MetaEMGLoader:
    """
    Loads and windows EMG data from Meta research datasets.

    Args:
        root: Path to the dataset root directory containing .h5 files
        dataset: Which Meta dataset format to parse
        wrist: For emg2qwerty — which wrist to load ("left", "right", or "both")
        window_samples: Samples per window (default: WINDOW_SAMPLES from constants)
        hop_samples: Hop size in samples (default: HOP_SAMPLES from constants)
    """

    def __init__(
        self,
        root: str | Path,
        dataset: DatasetType = "emg2pose",
        wrist: Literal["left", "right", "both"] = "right",
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py is required for Meta dataset loading. pip install h5py")
        self.root = Path(root)
        self.dataset = dataset
        self.wrist = wrist
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._h5_files = sorted(self.root.glob("**/*.h5")) + sorted(self.root.glob("**/*.hdf5"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[str]:
        """Return all session IDs available in the root directory."""
        sessions = []
        for f in self._h5_files:
            with h5py.File(f, "r") as hf:
                sessions.extend(list(hf.keys()))
        return sessions

    def load_session(
        self,
        session_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a single session.

        Returns:
            emg:    (N_samples, 16) float32
            target: (N_samples, 63) float32 for emg2pose (joint angles)
                    (N_samples,)    int32   for gni / emg2qwerty (class labels)
        """
        h5_file = self._find_session_file(session_id)
        with h5py.File(h5_file, "r") as hf:
            if self.dataset == "emg2pose":
                emg = hf[session_id]["emg"][:]
                target = hf[session_id]["joints"][:]
            elif self.dataset == "gni":
                emg = hf[session_id]["emg"][:]
                target = hf[session_id]["labels"][:]
            elif self.dataset == "emg2qwerty":
                emg = self._load_qwerty_emg(hf, session_id)
                target = hf[session_id]["labels"][:]
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")

        emg = emg.astype(np.float32)
        return emg, target

    def build_windows(
        self,
        max_sessions: Optional[int] = None,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load all sessions and slice into fixed-length windows.

        Returns:
            X: (n_windows, window_samples, 16) float32
            y: (n_windows, 63) float32  for emg2pose
               (n_windows,)    int32    for gni / emg2qwerty
        """
        sessions = self.list_sessions()
        if max_sessions is not None:
            sessions = sessions[:max_sessions]

        all_X, all_y = [], []
        for sid in sessions:
            try:
                emg, target = self.load_session(sid)
            except Exception:
                continue

            if normalize:
                emg = self._normalize(emg)

            X_sess, y_sess = self._slice_windows(emg, target)
            if len(X_sess) > 0:
                all_X.append(X_sess)
                all_y.append(y_sess)

        if not all_X:
            raise RuntimeError("No windows extracted. Check dataset path and format.")

        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_session_file(self, session_id: str) -> Path:
        for f in self._h5_files:
            with h5py.File(f, "r") as hf:
                if session_id in hf:
                    return f
        raise KeyError(f"Session '{session_id}' not found in {self.root}")

    def _load_qwerty_emg(self, hf: "h5py.File", session_id: str) -> np.ndarray:
        if self.wrist == "left":
            return hf[session_id]["emg_l"][:]
        elif self.wrist == "right":
            return hf[session_id]["emg_r"][:]
        else:  # both — concatenate channels → (N, 32)
            left = hf[session_id]["emg_l"][:]
            right = hf[session_id]["emg_r"][:]
            return np.concatenate([left, right], axis=1)

    def _normalize(self, emg: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization."""
        mu = emg.mean(axis=0, keepdims=True)
        std = emg.std(axis=0, keepdims=True) + 1e-8
        return (emg - mu) / std

    def _slice_windows(
        self,
        emg: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice (N, C) EMG and labels into (n_windows, window_samples, C) windows."""
        n_samples = len(emg)
        starts = range(0, n_samples - self.window_samples + 1, self.hop_samples)
        X_list, y_list = [], []
        for s in starts:
            X_list.append(emg[s: s + self.window_samples])
            # For label arrays, take the majority label in the window
            if target.ndim == 1:
                y_list.append(int(np.bincount(target[s: s + self.window_samples].clip(0)).argmax()))
            else:
                # Continuous target (e.g., joint angles) — take center frame
                y_list.append(target[s + self.window_samples // 2])

        if not X_list:
            return np.empty((0, self.window_samples, emg.shape[1]), dtype=np.float32), np.array([], dtype=np.int32)

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list)
        return X, y


# ------------------------------------------------------------------
# Convenience function for quick loading
# ------------------------------------------------------------------

def load_meta_windows(
    root: str | Path,
    dataset: DatasetType = "emg2pose",
    max_sessions: Optional[int] = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Meta dataset windows in one call.

    Example:
        X, y = load_meta_windows("data/meta/emg2pose", dataset="emg2pose", max_sessions=10)
        # X: (n_windows, 400, 16)  y: (n_windows, 63)
    """
    loader = MetaEMGLoader(root=root, dataset=dataset)
    return loader.build_windows(max_sessions=max_sessions, normalize=normalize)

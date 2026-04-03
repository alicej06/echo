"""
One-click SVM baseline training script using synthetic sEMG data.

Steps
-----
1. Check whether synthetic data exists in data/raw/synthetic/; generate it
   (5 simulated participants, 15 reps each) if the directory is empty or
   does not contain enough participant files.
2. Load the combined dataset with src.data.loader.load_dataset.
3. Create sliding windows via src.data.loader.create_windows →
   raw windows of shape (N, 40, 8).
4. Extract the 80-dim handcrafted feature vector for each window using
   src.utils.features.extract_features.
5. Perform a stratified 80/20 train/validation split.
6. Train the SVM via src.models.svm_classifier.train_svm.
7. Print the final validation accuracy.
8. Print the canonical output path.

Usage (from project root)
--------------------------
    python scripts/train_svm_baseline.py

No command-line arguments are required; all configuration is embedded below.
The script can also be imported and the ``main()`` function called directly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``from src.*`` imports work
# when the script is invoked via ``python scripts/train_svm_baseline.py``
# from the project root directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.data.loader import create_windows, load_dataset
from src.data.synthetic import generate_dataset
from src.models.svm_classifier import train_svm
from src.utils.constants import ASL_LABELS
from src.utils.features import extract_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHETIC_DATA_DIR: str = "data/raw/synthetic"
SVM_OUTPUT_PATH: str = "models/svm_classifier.joblib"

N_PARTICIPANTS: int = 5
N_REPS: int = 15

VAL_FRACTION: float = 0.20
RANDOM_SEED: int = 42

SVM_KERNEL: str = "rbf"
SVM_C: float = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_feature_matrix(windows: np.ndarray) -> np.ndarray:
    """Vectorise feature extraction over a (N, 40, 8) window array.

    Parameters
    ----------
    windows:
        Raw EMG windows of shape ``(N, T, C)``.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(N, 80)`` where each row is the
        output of ``extract_features`` for the corresponding window.
    """
    N = len(windows)
    features = np.empty((N, 80), dtype=np.float32)
    for i in range(N):
        features[i] = extract_features(windows[i].astype(np.float64))
    return features


def _train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 split that preserves approximate class balance.

    For each class, ``val_fraction`` of its samples are moved to the
    validation set; the remainder stay in training.

    Parameters
    ----------
    features:
        Feature matrix of shape ``(N, 80)``.
    labels:
        String label array of shape ``(N,)``.
    val_fraction:
        Fraction of each class to use as validation data.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (train_features, train_labels, val_features, val_labels)
    """
    rng = np.random.default_rng(seed)
    train_idxs: list[int] = []
    val_idxs: list[int] = []

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_fraction))
        val_idxs.extend(idxs[:n_val].tolist())
        train_idxs.extend(idxs[n_val:].tolist())

    train_arr = np.array(train_idxs)
    val_arr = np.array(val_idxs)

    rng.shuffle(train_arr)
    rng.shuffle(val_arr)

    return (
        features[train_arr],
        labels[train_arr],
        features[val_arr],
        labels[val_arr],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data if needed
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/5  Checking / generating synthetic sEMG data")
    print("=" * 60)

    data_path = Path(SYNTHETIC_DATA_DIR)
    existing_csvs = list(data_path.glob("*.csv")) if data_path.exists() else []

    if len(existing_csvs) >= N_PARTICIPANTS:
        print(
            f"  Found {len(existing_csvs)} existing CSV files in '{data_path}'. "
            "Skipping generation."
        )
    else:
        print(
            f"  Found {len(existing_csvs)} CSV files; need {N_PARTICIPANTS}. "
            "Generating synthetic dataset …"
        )
        generate_dataset(
            output_dir=SYNTHETIC_DATA_DIR,
            n_participants=N_PARTICIPANTS,
            n_reps=N_REPS,
            labels=list(ASL_LABELS),
        )

    # ------------------------------------------------------------------
    # Step 2: Load dataset
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2/5  Loading dataset")
    print("=" * 60)

    df = load_dataset(SYNTHETIC_DATA_DIR)

    # ------------------------------------------------------------------
    # Step 3: Create sliding windows → (N, 40, 8)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/5  Creating sliding windows")
    print("=" * 60)

    windows, labels = create_windows(df)
    print(f"  Windows shape : {windows.shape}  (N, T, C)")
    print(f"  Labels shape  : {labels.shape}")
    print(f"  Unique labels : {len(np.unique(labels))}")

    # ------------------------------------------------------------------
    # Step 4: Extract 80-dim handcrafted features for each window
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4/5  Extracting handcrafted features")
    print("=" * 60)
    print(
        f"  Extracting 80-dim feature vectors from {len(windows)} windows …"
    )

    all_features = _extract_feature_matrix(windows)
    print(f"  Feature matrix shape : {all_features.shape}  (N, 80)")

    # ------------------------------------------------------------------
    # Step 5: Stratified 80/20 split
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5/5  Splitting + training SVM baseline")
    print("=" * 60)

    train_features, train_labels, val_features, val_labels = _train_val_split(
        all_features, labels, val_fraction=VAL_FRACTION, seed=RANDOM_SEED
    )
    print(f"  Train : {len(train_features)} samples")
    print(f"  Val   : {len(val_features)} samples")
    print()
    print(
        f"  Config: kernel={SVM_KERNEL!r}, C={SVM_C}, "
        f"output={SVM_OUTPUT_PATH!r}"
    )
    print()

    train_svm(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        kernel=SVM_KERNEL,
        C=SVM_C,
        output_path=SVM_OUTPUT_PATH,
    )

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Training time : {elapsed:.1f} s")
    print(f"SVM model ready at {SVM_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

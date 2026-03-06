"""
One-click baseline training script using synthetic sEMG data.

Steps
-----
1. Generate synthetic data (5 simulated participants, 15 reps each) into
   data/raw/synthetic/.
2. Load the combined dataset with src.data.loader.load_dataset / create_windows.
3. Perform an 80/20 random train/validation split.
4. Train ASLEMGClassifier using src.models.train.train_model with:
       epochs=50, batch_size=64, hidden_size=128, lr=1e-3, wandb disabled.
5. Export the best checkpoint to ONNX: models/asl_emg_classifier.onnx.
6. Also save the PyTorch checkpoint to: models/asl_emg_classifier.pt.
7. Print final validation accuracy and the ONNX output path.

Usage (from project root)
--------------------------
    python scripts/train_synthetic_baseline.py

No command-line arguments are required; all configuration is embedded below.
The script can also be imported and the ``main()`` function called directly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``from src.*`` imports work
# when the script is invoked via ``python scripts/train_synthetic_baseline.py``
# from the project root directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.data.loader import create_windows, load_dataset
from src.data.synthetic import generate_dataset
from src.models.train import TrainConfig, train_model
from src.utils.constants import ASL_LABELS, NUM_CLASSES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHETIC_DATA_DIR: str = "data/raw/synthetic"
ONNX_OUTPUT_PATH: str = "models/asl_emg_classifier.onnx"
PT_OUTPUT_PATH: str = "models/asl_emg_classifier.pt"
CHECKPOINT_NAME: str = "best_model.pt"

N_PARTICIPANTS: int = 5
N_REPS: int = 15

TRAIN_CONFIG = TrainConfig(
    hidden_size=128,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=50,
    early_stopping_patience=10,
    lr_patience=5,
    lr_factor=0.5,
    lr_min=1e-6,
    checkpoint_dir="models",
    checkpoint_name=CHECKPOINT_NAME,
    device="cpu",
    wandb_project="",        # empty string → wandb disabled
    wandb_run_name="",
    label_names=list(ASL_LABELS),
)

VAL_FRACTION: float = 0.20
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_val_split(
    windows: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified 80/20 split that preserves approximate class balance.

    For each class, ``val_fraction`` of its windows are moved to the
    validation set; the remainder stay in training.  This avoids empty
    classes in either split when the class count is small.
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

    train_idxs_arr = np.array(train_idxs)
    val_idxs_arr = np.array(val_idxs)

    # Shuffle both splits
    rng.shuffle(train_idxs_arr)
    rng.shuffle(val_idxs_arr)

    return (
        windows[train_idxs_arr],
        labels[train_idxs_arr],
        windows[val_idxs_arr],
        labels[val_idxs_arr],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/5  Generating synthetic sEMG data")
    print("=" * 60)

    data_path = Path(SYNTHETIC_DATA_DIR)
    existing_csvs = list(data_path.glob("*.csv")) if data_path.exists() else []

    if len(existing_csvs) >= N_PARTICIPANTS:
        print(
            f"  Found {len(existing_csvs)} existing CSV files in '{data_path}'. "
            "Skipping generation."
        )
    else:
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
    # Step 3: Windowing
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/5  Creating sliding windows")
    print("=" * 60)

    # create_windows returns (N, 40, 8) raw windows and string labels.
    # train_model will flatten these to (N, 320) internally and set
    # input_size=320 on the model.
    windows, labels = create_windows(df)

    print(f"  Windows shape : {windows.shape}  (N, T, C)")
    print(f"  Labels shape  : {labels.shape}")
    print(f"  Unique labels : {len(np.unique(labels))}")

    # ------------------------------------------------------------------
    # Step 4: Train/validation split (stratified 80/20)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4/5  Splitting into train / val (80 / 20, stratified)")
    print("=" * 60)

    train_windows, train_labels, val_windows, val_labels = _train_val_split(
        windows, labels, val_fraction=VAL_FRACTION, seed=RANDOM_SEED
    )

    print(f"  Train : {len(train_windows)} windows")
    print(f"  Val   : {len(val_windows)} windows")

    # ------------------------------------------------------------------
    # Step 5: Train model
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5/5  Training ASLEMGClassifier")
    print("=" * 60)
    print(
        f"  Config: epochs={TRAIN_CONFIG.epochs}, "
        f"batch_size={TRAIN_CONFIG.batch_size}, "
        f"hidden_size={TRAIN_CONFIG.hidden_size}, "
        f"lr={TRAIN_CONFIG.lr}"
    )
    print()

    model = train_model(
        train_windows=train_windows,
        train_labels=train_labels,
        val_windows=val_windows,
        val_labels=val_labels,
        config=TRAIN_CONFIG,
    )

    # ------------------------------------------------------------------
    # Compute final validation accuracy
    # ------------------------------------------------------------------
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()

    # Flatten validation windows the same way train_model does
    N_val, T, C = val_windows.shape
    val_flat = val_windows.reshape(N_val, T * C).astype(np.float32)

    label_map = {name: i for i, name in enumerate(ASL_LABELS)}
    val_y_int = np.array([label_map[l] for l in val_labels], dtype=np.int64)

    X_tensor = torch.from_numpy(val_flat)
    y_tensor = torch.from_numpy(val_y_int)
    val_ds = TensorDataset(X_tensor, y_tensor)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    final_val_acc = correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Export PyTorch checkpoint to the canonical path
    # ------------------------------------------------------------------
    pt_path = Path(PT_OUTPUT_PATH)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(pt_path)
    print(f"\nPyTorch model saved to {pt_path}")

    # ------------------------------------------------------------------
    # Export ONNX
    # ------------------------------------------------------------------
    onnx_path = Path(ONNX_OUTPUT_PATH)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.to_onnx(onnx_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Final validation accuracy : {final_val_acc:.4f}  ({final_val_acc * 100:.2f} %)")
    print(f"Training time             : {elapsed:.1f} s")
    print(f"Model ready at {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

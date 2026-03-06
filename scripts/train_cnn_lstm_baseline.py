"""
One-click baseline training script for the CNN-LSTM classifier using synthetic
sEMG data.

Unlike train_synthetic_baseline.py (which feeds hand-crafted 80-dim feature
vectors into the LSTM), this script passes **raw windows of shape (N, 40, 8)**
directly to CNNLSTMClassifier.  The CNN front-end learns spatial and
local-temporal features; the LSTM models longer-range dynamics over the
down-sampled representation.

Steps
-----
1. Generate synthetic data (5 simulated participants, 15 reps each) into
   data/raw/synthetic/ — skipped if enough CSVs already exist.
2. Load the combined dataset with src.data.loader.load_dataset / create_windows.
   Windows come out as (N, 40, 8) — no flattening performed here.
3. Stratified 80/20 train/val split.
4. Train CNNLSTMClassifier with an inline training loop:
       Adam lr=1e-3, weight_decay=1e-4
       ReduceLROnPlateau patience=5
       epochs=50, early-stopping patience=10
       batch_size=64
5. Export the best checkpoint to:
       models/cnn_lstm_classifier.onnx
       models/cnn_lstm_classifier.pt
6. Print final validation accuracy and the ONNX output path.

Usage (from project root)
--------------------------
    python scripts/train_cnn_lstm_baseline.py

No command-line arguments are required.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src.* imports resolve when called as ``python scripts/<name>.py``
# from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import create_windows, load_dataset
from src.data.synthetic import generate_dataset
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.utils.constants import ASL_LABELS, NUM_CLASSES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHETIC_DATA_DIR: str = "data/raw/synthetic"
ONNX_OUTPUT_PATH: str = "models/cnn_lstm_classifier.onnx"
PT_OUTPUT_PATH: str = "models/cnn_lstm_classifier.pt"
CHECKPOINT_PATH: str = "models/cnn_lstm_best.pt"

N_PARTICIPANTS: int = 5
N_REPS: int = 15

# Training hyper-parameters
DROPOUT: float = 0.3
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64
EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 10
LR_PATIENCE: int = 5
LR_FACTOR: float = 0.5
LR_MIN: float = 1e-6

VAL_FRACTION: float = 0.20
RANDOM_SEED: int = 42

DEVICE: str = "cpu"


# ---------------------------------------------------------------------------
# Stratified 80/20 split  (same logic as train_synthetic_baseline.py)
# ---------------------------------------------------------------------------


def _train_val_split(
    windows: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split that preserves approximate per-class proportions.

    For each class ``val_fraction`` of its windows go to validation; the
    rest go to training.  At least one window per class is always held out.
    """
    rng = np.random.default_rng(seed)
    train_idxs: list[int] = []
    val_idxs: list[int] = []

    for lbl in np.unique(labels):
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
        windows[train_arr],
        labels[train_arr],
        windows[val_arr],
        labels[val_arr],
    )


# ---------------------------------------------------------------------------
# Label encoding helper
# ---------------------------------------------------------------------------


def _encode_labels(labels: np.ndarray, label_names: list[str]) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(label_names)}
    try:
        return np.array([mapping[l] for l in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"Label '{exc.args[0]}' not found in label_names."
        ) from exc


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def _make_dataloader(
    windows: np.ndarray,
    labels_int: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader from (N, T, C) windows and integer labels."""
    X = torch.from_numpy(windows.astype(np.float32))   # (N, 40, 8)
    y = torch.from_numpy(labels_int)
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Inline training loop for CNNLSTMClassifier
# ---------------------------------------------------------------------------


def _train_cnn_lstm(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    label_names: list[str],
    checkpoint_path: str | Path,
) -> CNNLSTMClassifier:
    """
    Train a :class:`CNNLSTMClassifier` on raw (N, 40, 8) windows.

    Parameters
    ----------
    train_windows, val_windows : np.ndarray
        Shape ``(N, 40, 8)`` — NOT flattened.
    train_labels, val_labels : np.ndarray
        String label arrays of shape ``(N,)``.
    label_names : list[str]
        Ordered list of class names (index == class id).
    checkpoint_path : str or Path
        Where to save the best model weights during training.

    Returns
    -------
    CNNLSTMClassifier
        Best model (by validation loss) loaded from *checkpoint_path*.
    """
    device = torch.device(DEVICE)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode string labels → integers
    train_y = _encode_labels(train_labels, label_names)
    val_y = _encode_labels(val_labels, label_names)

    train_loader = _make_dataloader(train_windows, train_y, BATCH_SIZE, shuffle=True)
    val_loader = _make_dataloader(val_windows, val_y, BATCH_SIZE, shuffle=False)

    model = CNNLSTMClassifier(
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        label_names=label_names,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=LR_PATIENCE,
        factor=LR_FACTOR,
        min_lr=LR_MIN,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)   # (B, 40, 8)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)        # (B, num_classes)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ---- Validate ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={current_lr:.2e}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            model.save(checkpoint_path)
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"[early stopping] No improvement for {no_improve} epochs. "
                f"Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f})"
            )
            break

    print(
        f"\n[_train_cnn_lstm] Training complete. "
        f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
    )

    # Reload the best checkpoint
    best_model = CNNLSTMClassifier.load(checkpoint_path)
    return best_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data if needed
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/5  Generating synthetic sEMG data")
    print("=" * 60)

    data_path = Path(SYNTHETIC_DATA_DIR)
    existing_csvs = list(data_path.glob("*.csv")) if data_path.exists() else []

    if len(existing_csvs) >= N_PARTICIPANTS:
        print(
            f"  Found {len(existing_csvs)} existing CSV file(s) in '{data_path}'. "
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
    # Step 3: Sliding-window segmentation  → (N, 40, 8) raw windows
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/5  Creating sliding windows  (no feature extraction)")
    print("=" * 60)

    windows, labels = create_windows(df)
    # windows.shape == (N, WINDOW_SIZE_SAMPLES=40, N_CHANNELS=8)

    print(f"  Windows shape : {windows.shape}  (N, T, C)")
    print(f"  Labels shape  : {labels.shape}")
    print(f"  Unique labels : {len(np.unique(labels))}")

    # ------------------------------------------------------------------
    # Step 4: Stratified 80/20 split
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
    # Step 5: Train CNN-LSTM
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5/5  Training CNNLSTMClassifier")
    print("=" * 60)
    print(
        f"  Config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"lr={LR}, weight_decay={WEIGHT_DECAY}, dropout={DROPOUT}, "
        f"early_stopping_patience={EARLY_STOPPING_PATIENCE}"
    )
    print()

    model = _train_cnn_lstm(
        train_windows=train_windows,
        train_labels=train_labels,
        val_windows=val_windows,
        val_labels=val_labels,
        label_names=list(ASL_LABELS),
        checkpoint_path=CHECKPOINT_PATH,
    )

    # ------------------------------------------------------------------
    # Final validation accuracy
    # ------------------------------------------------------------------
    device = torch.device(DEVICE)
    model.eval()
    model.to(device)

    label_map = {name: i for i, name in enumerate(ASL_LABELS)}
    val_y_int = np.array([label_map[l] for l in val_labels], dtype=np.int64)

    X_tensor = torch.from_numpy(val_windows.astype(np.float32))   # (N, 40, 8)
    y_tensor = torch.from_numpy(val_y_int)
    val_loader_final = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=128,
        shuffle=False,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_final:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    final_val_acc = correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Export PyTorch checkpoint to canonical path
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
    print(
        f"Final validation accuracy : {final_val_acc:.4f}  "
        f"({final_val_acc * 100:.2f} %)"
    )
    print(f"Training time             : {elapsed:.1f} s")
    print(f"CNN-LSTM model ready at {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

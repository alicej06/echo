"""
One-click baseline training script for the Conformer classifier using synthetic
sEMG data.

Mirrors train_cnn_lstm_baseline.py in structure but trains ConformerClassifier,
which operates on raw windows of shape (N, 40, 8) and uses a Conv-Transformer
architecture rather than a recurrent model.

Steps
-----
1. Generate synthetic data (5 simulated participants, 15 reps each) into
   data/raw/synthetic/ -- skipped if enough CSVs already exist.
2. Load the combined dataset with src.data.loader.load_dataset / create_windows.
   Windows come out as (N, 40, 8) -- no flattening performed here.
3. Stratified 80/20 train/val split.
4. Train ConformerClassifier with an inline training loop:
       Adam lr=1e-3, weight_decay=1e-4
       ReduceLROnPlateau patience=5
       epochs=100, early-stopping patience=15
       batch_size=64
5. Export the best checkpoint to:
       models/conformer_classifier.pt
       models/conformer_classifier.onnx
6. Print final validation accuracy, inference latency, and a comparison
   against the CNN-LSTM baseline (if its checkpoint exists).

Usage (from project root)
--------------------------
    python scripts/train_conformer_baseline.py

No command-line arguments are required.
"""

from __future__ import annotations

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
from src.models.conformer_classifier import ConformerClassifier
from src.utils.constants import ASL_LABELS, NUM_CLASSES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHETIC_DATA_DIR: str = "data/raw/synthetic"
ONNX_OUTPUT_PATH: str = "models/conformer_classifier.onnx"
PT_OUTPUT_PATH: str = "models/conformer_classifier.pt"
CHECKPOINT_PATH: str = "models/conformer_best.pt"

# CNN-LSTM checkpoint path -- used only for the optional comparison at the end.
CNNLSTM_PT_PATH: str = "models/cnn_lstm_classifier.pt"

N_PARTICIPANTS: int = 5
N_REPS: int = 15

# Training hyper-parameters
D_MODEL: int = 128
N_LAYERS: int = 4
N_HEADS: int = 4
CONV_KERNEL: int = 15
DROPOUT: float = 0.1
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 64
EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 15
LR_PATIENCE: int = 5
LR_FACTOR: float = 0.5
LR_MIN: float = 1e-6

VAL_FRACTION: float = 0.20
RANDOM_SEED: int = 42

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Stratified 80/20 split
# ---------------------------------------------------------------------------


def _train_val_split(
    windows: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split preserving approximate per-class proportions.

    For each class, ``val_fraction`` of its windows go to validation; the
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
# Label encoding
# ---------------------------------------------------------------------------


def _encode_labels(labels: np.ndarray, label_names: list[str]) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(label_names)}
    try:
        return np.array([mapping[lbl] for lbl in labels], dtype=np.int64)
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
# Inline training loop
# ---------------------------------------------------------------------------


def _train_conformer(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    label_names: list[str],
    checkpoint_path: str | Path,
) -> ConformerClassifier:
    """Train a ConformerClassifier on raw (N, 40, 8) windows.

    Parameters
    ----------
    train_windows, val_windows : np.ndarray
        Shape (N, 40, 8) -- NOT flattened.
    train_labels, val_labels : np.ndarray
        String label arrays of shape (N,).
    label_names : list[str]
        Ordered list of class names (index == class id).
    checkpoint_path : str or Path
        Where to save the best model weights during training.

    Returns
    -------
    ConformerClassifier
        Best model (by validation loss) loaded from checkpoint_path.
    """
    device = torch.device(DEVICE)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_y = _encode_labels(train_labels, label_names)
    val_y = _encode_labels(val_labels, label_names)

    train_loader = _make_dataloader(train_windows, train_y, BATCH_SIZE, shuffle=True)
    val_loader = _make_dataloader(val_windows, val_y, BATCH_SIZE, shuffle=False)

    model = ConformerClassifier(
        n_classes=NUM_CLASSES,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        conv_kernel=CONV_KERNEL,
        dropout=DROPOUT,
        label_names=label_names,
    ).to(device)

    print(f"  Model: {model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")
    print()

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
            logits = model(X_batch)        # (B, n_classes)
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
        f"\n[_train_conformer] Training complete. "
        f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}."
    )

    best_model = ConformerClassifier.load(checkpoint_path)
    return best_model


# ---------------------------------------------------------------------------
# Inference latency measurement
# ---------------------------------------------------------------------------


def _measure_latency_ms(
    model: nn.Module,
    windows: np.ndarray,
    n_repeats: int = 10,
) -> float:
    """Return mean per-sample inference latency in milliseconds.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode on CPU (latency comparison is CPU-side).
    windows : np.ndarray
        Shape (N, T, C) sample batch.
    n_repeats : int
        Number of timed passes to average over.
    """
    model.eval()
    X = torch.from_numpy(windows.astype(np.float32))
    N = len(windows)
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model(X)
            times.append((time.perf_counter() - t0) * 1000.0 / N)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data if needed
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/6  Generating synthetic sEMG data")
    print("=" * 60)

    data_path = Path(SYNTHETIC_DATA_DIR)
    existing_csvs = list(data_path.glob("*.csv")) if data_path.exists() else []

    if len(existing_csvs) >= N_PARTICIPANTS:
        print(
            f"  Found {len(existing_csvs)} existing CSV file(s) in '{data_path}'. "
            "Skipping generation."
        )
    else:
        print(
            f"  Generating {N_PARTICIPANTS} synthetic participants "
            f"({N_REPS} reps each) ..."
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
    print("Step 2/6  Loading dataset")
    print("=" * 60)

    df = load_dataset(SYNTHETIC_DATA_DIR)

    # ------------------------------------------------------------------
    # Step 3: Sliding-window segmentation  -> (N, 40, 8) raw windows
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/6  Creating sliding windows  (no feature extraction)")
    print("=" * 60)

    windows, labels = create_windows(df)
    print(f"  Windows shape : {windows.shape}  (N, T, C)")
    print(f"  Labels shape  : {labels.shape}")
    print(f"  Unique labels : {len(np.unique(labels))}")

    # ------------------------------------------------------------------
    # Step 4: Stratified 80/20 split
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4/6  Splitting into train / val (80 / 20, stratified)")
    print("=" * 60)

    train_windows, train_labels, val_windows, val_labels = _train_val_split(
        windows, labels, val_fraction=VAL_FRACTION, seed=RANDOM_SEED
    )
    print(f"  Train : {len(train_windows)} windows")
    print(f"  Val   : {len(val_windows)} windows")

    # ------------------------------------------------------------------
    # Step 5: Train Conformer
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5/6  Training ConformerClassifier")
    print("=" * 60)
    print(
        f"  Config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
        f"lr={LR}, weight_decay={WEIGHT_DECAY}, dropout={DROPOUT}, "
        f"d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}, "
        f"conv_kernel={CONV_KERNEL}, "
        f"early_stopping_patience={EARLY_STOPPING_PATIENCE}"
    )
    print()

    t_train_start = time.time()
    model = _train_conformer(
        train_windows=train_windows,
        train_labels=train_labels,
        val_windows=val_windows,
        val_labels=val_labels,
        label_names=list(ASL_LABELS),
        checkpoint_path=CHECKPOINT_PATH,
    )
    train_elapsed = time.time() - t_train_start

    # ------------------------------------------------------------------
    # Final validation accuracy
    # ------------------------------------------------------------------
    device_cpu = torch.device("cpu")
    model.eval()
    model.to(device_cpu)

    label_map = {name: i for i, name in enumerate(ASL_LABELS)}
    val_y_int = np.array([label_map[lbl] for lbl in val_labels], dtype=np.int64)

    X_val = torch.from_numpy(val_windows.astype(np.float32))
    y_val = torch.from_numpy(val_y_int)
    val_loader_final = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=128,
        shuffle=False,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_final:
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    final_val_acc = correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Inference latency (Conformer, CPU)
    # ------------------------------------------------------------------
    sample_windows = val_windows[:100]
    conformer_latency_ms = _measure_latency_ms(model, sample_windows)

    # ------------------------------------------------------------------
    # Export PyTorch checkpoint
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
    # Step 6: Comparison vs CNN-LSTM (if checkpoint available)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6/6  Accuracy and latency comparison")
    print("=" * 60)

    cnnlstm_latency_ms: float | None = None
    cnnlstm_val_acc: float | None = None

    cnnlstm_path = Path(CNNLSTM_PT_PATH)
    if cnnlstm_path.exists():
        try:
            from src.models.cnn_lstm_classifier import CNNLSTMClassifier

            cnnlstm = CNNLSTMClassifier.load(cnnlstm_path)
            cnnlstm.eval()

            # CNN-LSTM latency on same 100-window sample
            cnnlstm_latency_ms = _measure_latency_ms(cnnlstm, sample_windows)

            # CNN-LSTM val accuracy on the same split
            cnnlstm_correct = 0
            cnnlstm_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader_final:
                    preds = cnnlstm(X_batch).argmax(dim=1)
                    cnnlstm_correct += (preds == y_batch).sum().item()
                    cnnlstm_total += len(y_batch)
            cnnlstm_val_acc = (
                cnnlstm_correct / cnnlstm_total if cnnlstm_total > 0 else 0.0
            )
        except Exception as exc:
            print(f"  [warning] Could not load CNN-LSTM for comparison: {exc}")

    # Comparison table
    col = 24
    sep = "-" * (col + 1) + "|" + "-" * (col - 1) + "|" + "-" * (col + 2)
    print()
    print(
        f"{'Model':>{col}} | {'Val Accuracy':>{col - 2}} | {'Latency (ms/sample)':>{col}}"
    )
    print(sep)
    print(
        f"{'Conformer':>{col}} | "
        f"{final_val_acc:>{col - 2}.4f} | "
        f"{conformer_latency_ms:>{col}.3f}"
    )
    if cnnlstm_val_acc is not None and cnnlstm_latency_ms is not None:
        print(
            f"{'CNN-LSTM':>{col}} | "
            f"{cnnlstm_val_acc:>{col - 2}.4f} | "
            f"{cnnlstm_latency_ms:>{col}.3f}"
        )
        delta_acc = final_val_acc - cnnlstm_val_acc
        sign = "+" if delta_acc >= 0 else ""
        print(
            f"\n  Conformer vs CNN-LSTM acc delta: {sign}{delta_acc:.4f}"
        )
    else:
        print(
            f"\n  (Run train_cnn_lstm_baseline.py first to enable comparison.)"
        )

    print()
    elapsed = time.time() - t_start
    print("=" * 60)
    print(
        f"Final validation accuracy : {final_val_acc:.4f}  "
        f"({final_val_acc * 100:.2f} %)"
    )
    print(f"Training time             : {train_elapsed:.1f} s")
    print(f"Total wall time           : {elapsed:.1f} s")
    print(f"Conformer model ready at  : {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

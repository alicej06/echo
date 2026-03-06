"""
Benchmark all available EMG-ASL model types on the same dataset.

Compares:
  - LSTM         : ASLEMGClassifier on flattened raw windows (N, 40*8 = 320)
  - CNN-LSTM     : CNNLSTMClassifier on raw windows (N, 40, 8)
  - SVM          : sklearn SVM on 80-dim feature vectors
  - Conformer    : ConformerClassifier on raw windows (N, 40, 8)

If --data-dir doesn't exist or is empty, synthetic data is generated
automatically into data/raw/synthetic/ and used.

Usage (from project root)
--------------------------
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --data-dir data/raw/synthetic/
    python scripts/benchmark_models.py --data-dir data/raw/ --epochs 30

Comparison table printed at end:

  Model        | Train Acc | Val Acc | Train Time | Inference (ms/sample)
  -------------|-----------|---------|------------|----------------------
  LSTM         |   0.XX    |  0.XX   |   XX.Xs    |       X.Xms
  CNN-LSTM     |   0.XX    |  0.XX   |   XX.Xs    |       X.Xms
  SVM          |   0.XX    |  0.XX   |   XX.Xs    |       X.Xms
  Conformer    |   0.XX    |  0.XX   |   XX.Xs    |       X.Xms
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.*`` imports work when
# invoked as ``python scripts/benchmark_models.py`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import create_windows, load_dataset
from src.data.synthetic import generate_dataset
from src.models.conformer_classifier import ConformerClassifier
from src.models.train import TrainConfig, train_model
from src.utils.constants import (
    ASL_LABELS,
    FEATURE_VECTOR_SIZE,
    N_CHANNELS,
    NUM_CLASSES,
    SAMPLE_RATE,
    WINDOW_SIZE_SAMPLES,
)
from src.utils.features import extract_features

# ---------------------------------------------------------------------------
# CNN-LSTM Classifier
# ---------------------------------------------------------------------------
# Imported here; the class is defined below as a local model since
# CNNLSTMClassifier does not yet have a canonical src/ module location.
# ---------------------------------------------------------------------------


class CNNLSTMClassifier(nn.Module):
    """
    Lightweight CNN-LSTM model for EMG-ASL classification.

    Architecture:
      Conv1D(in=8, out=32, k=3, padding=1) + ReLU + BatchNorm + MaxPool(2)
      Conv1D(in=32, out=64, k=3, padding=1) + ReLU + BatchNorm + MaxPool(2)
      LSTM(64, 128, num_layers=1, batch_first=True)
      Dropout(0.3)
      Linear(128, num_classes)

    Input shape: (batch, seq_len, n_channels) = (B, 40, 8)
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        seq_len: int = WINDOW_SIZE_SAMPLES,
        num_classes: int = NUM_CLASSES,
        cnn_channels: Tuple[int, int] = (32, 64),
        lstm_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1, c2 = cnn_channels

        # Conv layers expect (B, C_in, L) — we'll permute inside forward.
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(c2),
            nn.MaxPool1d(kernel_size=2),
        )

        # After two MaxPool(2): seq_len -> seq_len // 4
        lstm_seq = seq_len // 4
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_channels)
        Returns
        -------
        logits : (batch, num_classes)
        """
        # (B, T, C) -> (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)   # (B, c1, T/2)
        x = self.conv2(x)   # (B, c2, T/4)
        # (B, c2, T/4) -> (B, T/4, c2) for LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)          # (B, T/4, hidden)
        last = lstm_out[:, -1, :]           # (B, hidden)
        out = self.dropout(last)
        return self.fc(out)                 # (B, num_classes)


# ---------------------------------------------------------------------------
# Stratified 80/20 split helper
# ---------------------------------------------------------------------------


def _stratified_split(
    windows: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 split preserving class balance."""
    rng = np.random.default_rng(seed)
    train_idxs: List[int] = []
    val_idxs: List[int] = []

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
        windows[train_arr], labels[train_arr],
        windows[val_arr],   labels[val_arr],
    )


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    mapping = {name: i for i, name in enumerate(ASL_LABELS)}
    return np.array([mapping[l] for l in labels], dtype=np.int64)


# ---------------------------------------------------------------------------
# LSTM benchmark
# ---------------------------------------------------------------------------


def _bench_lstm(
    train_w: np.ndarray,
    train_l: np.ndarray,
    val_w: np.ndarray,
    val_l: np.ndarray,
    epochs: int,
) -> Dict:
    """Train and evaluate the standard LSTM (ASLEMGClassifier) via train_model."""
    config = TrainConfig(
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        epochs=epochs,
        early_stopping_patience=10,
        lr_patience=5,
        lr_factor=0.5,
        lr_min=1e-6,
        checkpoint_dir="models",
        checkpoint_name="bench_lstm.pt",
        device="cpu",
        wandb_project="",
        label_names=list(ASL_LABELS),
    )

    t0 = time.perf_counter()
    model = train_model(
        train_windows=train_w,
        train_labels=train_l,
        val_windows=val_w,
        val_labels=val_l,
        config=config,
    )
    train_time = time.perf_counter() - t0

    # Compute training accuracy
    model.eval()
    device = torch.device("cpu")
    N, T, C = train_w.shape
    train_flat = train_w.reshape(N, T * C).astype(np.float32)
    train_y = _encode_labels(train_l)
    train_acc = _torch_accuracy(model, train_flat, train_y, device)

    # Val accuracy
    val_flat = val_w.reshape(val_w.shape[0], T * C).astype(np.float32)
    val_y = _encode_labels(val_l)
    val_acc = _torch_accuracy(model, val_flat, val_y, device)

    # Inference timing (ms per sample)
    infer_ms = _torch_infer_time(model, train_flat[:100], device)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_time": train_time,
        "infer_ms": infer_ms,
    }


def _torch_accuracy(
    model: nn.Module,
    X_flat: np.ndarray,
    y_int: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    model.eval()
    X_t = torch.from_numpy(X_flat)
    y_t = torch.from_numpy(y_int)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            preds = model(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
    return correct / total if total > 0 else 0.0


def _torch_infer_time(
    model: nn.Module,
    X_flat: np.ndarray,
    device: torch.device,
    n_repeats: int = 5,
) -> float:
    """Return mean inference time in ms/sample over n_repeats passes."""
    model.eval()
    X_t = torch.from_numpy(X_flat.astype(np.float32)).to(device)
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model(X_t)
            times.append((time.perf_counter() - t0) * 1000.0 / len(X_flat))
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# CNN-LSTM benchmark
# ---------------------------------------------------------------------------


def _bench_cnn_lstm(
    train_w: np.ndarray,
    train_l: np.ndarray,
    val_w: np.ndarray,
    val_l: np.ndarray,
    epochs: int,
) -> Dict:
    """Train and evaluate CNNLSTMClassifier on raw 3-D windows."""
    device = torch.device("cpu")
    model = CNNLSTMClassifier(
        n_channels=N_CHANNELS,
        seq_len=WINDOW_SIZE_SAMPLES,
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    train_y = _encode_labels(train_l)
    val_y = _encode_labels(val_l)

    # DataLoaders: windows are (N, T, C) already, model permutes inside forward.
    X_train_t = torch.from_numpy(train_w.astype(np.float32))
    y_train_t = torch.from_numpy(train_y)
    X_val_t = torch.from_numpy(val_w.astype(np.float32))
    y_val_t = torch.from_numpy(val_y)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    best_val_loss = float("inf")
    no_improve = 0
    early_stopping_patience = 10

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(y_b)
            train_correct += (logits.argmax(dim=1) == y_b).sum().item()
            train_total += len(y_b)

        train_loss = train_loss_sum / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device)
                logits = model(X_b)
                val_loss_sum += criterion(logits, y_b).item() * len(y_b)
                val_correct += (logits.argmax(dim=1) == y_b).sum().item()
                val_total += len(y_b)

        val_loss = val_loss_sum / val_total
        scheduler.step(val_loss)

        print(
            f"  [CNN-LSTM] Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stopping_patience:
            print(f"  [CNN-LSTM] Early stopping at epoch {epoch}.")
            break

    train_time = time.perf_counter() - t0

    # Accuracy
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    # Inference timing (ms per sample) on val set
    model.eval()
    sample = val_w[:100].astype(np.float32)
    X_s = torch.from_numpy(sample).to(device)
    times = []
    with torch.no_grad():
        for _ in range(5):
            t_s = time.perf_counter()
            _ = model(X_s)
            times.append((time.perf_counter() - t_s) * 1000.0 / len(sample))
    infer_ms = float(np.mean(times))

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_time": train_time,
        "infer_ms": infer_ms,
    }


# ---------------------------------------------------------------------------
# Conformer benchmark
# ---------------------------------------------------------------------------


def _bench_conformer(
    train_w: np.ndarray,
    train_l: np.ndarray,
    val_w: np.ndarray,
    val_l: np.ndarray,
    epochs: int,
) -> Dict:
    """Train and evaluate ConformerClassifier on raw 3-D windows.

    Input shape to the model: (batch, T=40, C=8) -- identical to CNN-LSTM.
    """
    device = torch.device("cpu")
    model = ConformerClassifier(
        n_classes=NUM_CLASSES,
        label_names=list(ASL_LABELS),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )

    train_y = _encode_labels(train_l)
    val_y = _encode_labels(val_l)

    X_train_t = torch.from_numpy(train_w.astype(np.float32))
    y_train_t = torch.from_numpy(train_y)
    X_val_t = torch.from_numpy(val_w.astype(np.float32))
    y_val_t = torch.from_numpy(val_y)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    best_val_loss = float("inf")
    no_improve = 0
    early_stopping_patience = 10

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(y_b)
            train_correct += (logits.argmax(dim=1) == y_b).sum().item()
            train_total += len(y_b)

        train_loss = train_loss_sum / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device)
                logits = model(X_b)
                val_loss_sum += criterion(logits, y_b).item() * len(y_b)
                val_correct += (logits.argmax(dim=1) == y_b).sum().item()
                val_total += len(y_b)

        val_loss = val_loss_sum / val_total
        scheduler.step(val_loss)

        print(
            f"  [Conformer] Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stopping_patience:
            print(f"  [Conformer] Early stopping at epoch {epoch}.")
            break

    train_time = time.perf_counter() - t0

    train_acc = train_correct / train_total if train_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    # Inference timing (ms per sample) -- shape (N, 40, 8)
    model.eval()
    sample = val_w[:100].astype(np.float32)
    X_s = torch.from_numpy(sample).to(device)
    times = []
    with torch.no_grad():
        for _ in range(5):
            t_s = time.perf_counter()
            _ = model(X_s)
            times.append((time.perf_counter() - t_s) * 1000.0 / len(sample))
    infer_ms = float(np.mean(times))

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_time": train_time,
        "infer_ms": infer_ms,
    }


# ---------------------------------------------------------------------------
# SVM benchmark
# ---------------------------------------------------------------------------


def _extract_feature_matrix(windows: np.ndarray) -> np.ndarray:
    """Extract 80-dim feature vectors from (N, T, C) windows."""
    feats = np.zeros((len(windows), FEATURE_VECTOR_SIZE), dtype=np.float32)
    for i, w in enumerate(windows):
        feats[i] = extract_features(w.astype(np.float64), fs=float(SAMPLE_RATE))
    return feats


def _bench_svm(
    train_w: np.ndarray,
    train_l: np.ndarray,
    val_w: np.ndarray,
    val_l: np.ndarray,
) -> Dict:
    """Extract 80-dim features and train a LinearSVC."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    print("  [SVM] Extracting features from training windows …")
    t_feat = time.perf_counter()
    X_train = _extract_feature_matrix(train_w)
    X_val = _extract_feature_matrix(val_w)
    feat_time = time.perf_counter() - t_feat
    print(f"  [SVM] Feature extraction: {feat_time:.1f} s  shape={X_train.shape}")

    # Standardise
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    print("  [SVM] Fitting LinearSVC …")
    t0 = time.perf_counter()
    clf = LinearSVC(max_iter=5000, C=1.0)
    clf.fit(X_train_sc, train_l)
    train_time = time.perf_counter() - t0
    print(f"  [SVM] Fit time: {train_time:.1f} s")

    train_acc = float(np.mean(clf.predict(X_train_sc) == train_l))
    val_acc = float(np.mean(clf.predict(X_val_sc) == val_l))

    # Inference timing (ms per sample)
    sample = X_val_sc[:100]
    times = []
    for _ in range(5):
        t_s = time.perf_counter()
        _ = clf.predict(sample)
        times.append((time.perf_counter() - t_s) * 1000.0 / len(sample))
    infer_ms = float(np.mean(times))

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_time": train_time + feat_time,
        "infer_ms": infer_ms,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_table(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison table."""
    col_w = 12
    sep = "-" * (col_w + 1) + "|" + "-" * (col_w - 1) + "|" + "-" * (col_w - 1) + "|" + "-" * (col_w + 2) + "|" + "-" * (col_w + 3)
    header = (
        f"{'Model':>{col_w}} | {'Train Acc':>{col_w - 2}} | {'Val Acc':>{col_w - 2}} | "
        f"{'Train Time':>{col_w}} | {'Inference (ms/sample)':>{col_w + 2}}"
    )
    print()
    print(header)
    print(sep)
    for model_name, r in results.items():
        train_time_str = f"{r['train_time']:.1f}s"
        infer_str = f"{r['infer_ms']:.2f}ms"
        line = (
            f"{model_name:>{col_w}} | "
            f"{r['train_acc']:>{col_w - 2}.4f} | "
            f"{r['val_acc']:>{col_w - 2}.4f} | "
            f"{train_time_str:>{col_w}} | "
            f"{infer_str:>{col_w + 2}}"
        )
        print(line)
    print()


def _print_winner(results: Dict[str, Dict]) -> None:
    """Print which model won and a brief explanation."""
    best_val = max(results.items(), key=lambda kv: kv[1]["val_acc"])
    fastest_infer = min(results.items(), key=lambda kv: kv[1]["infer_ms"])
    fastest_train = min(results.items(), key=lambda kv: kv[1]["train_time"])

    print("=" * 60)
    print("BENCHMARK CONCLUSIONS")
    print("=" * 60)
    print(
        f"  Best val accuracy  : {best_val[0]}  "
        f"({best_val[1]['val_acc']:.4f})"
    )
    print(
        f"  Fastest inference  : {fastest_infer[0]}  "
        f"({fastest_infer[1]['infer_ms']:.2f} ms/sample)"
    )
    print(
        f"  Fastest training   : {fastest_train[0]}  "
        f"({fastest_train[1]['train_time']:.1f} s)"
    )
    print()

    winner = best_val[0]
    val_accs = {k: v["val_acc"] for k, v in results.items()}
    infer_mss = {k: v["infer_ms"] for k, v in results.items()}

    if winner == "LSTM":
        print(
            "LSTM wins on accuracy. It is also the simplest architecture and "
            "typically generalises well on small EMG datasets."
        )
    elif winner == "CNN-LSTM":
        print(
            "CNN-LSTM wins on accuracy. The convolutional front-end extracts "
            "local temporal patterns before the LSTM models long-range dependencies, "
            "making it more expressive for complex gestures."
        )
    elif winner == "SVM":
        print(
            "SVM wins on accuracy. Hand-crafted time- and frequency-domain features "
            "are highly discriminative on this dataset, and LinearSVC trains orders "
            "of magnitude faster than the deep models. Consider SVM as the baseline "
            "when data is limited."
        )
    elif winner == "Conformer":
        print(
            "Conformer wins on accuracy. The depthwise convolution captures motor "
            "unit potential morphology while multi-head self-attention models "
            "cross-channel muscle synergies across the full 200 ms window. "
            "Consider Conformer as the default architecture for GPU training on "
            "longer windows (400-800 ms) where recurrent models degrade."
        )

    # Trade-off note
    deep_models = ["LSTM", "CNN-LSTM", "Conformer"]
    fastest_deep = min(
        ((k, v["infer_ms"]) for k, v in results.items() if k in deep_models),
        key=lambda kv: kv[1],
        default=(None, float("inf")),
    )
    if infer_mss.get("SVM", float("inf")) < infer_mss.get("LSTM", float("inf")):
        print(
            "\nFor real-time BLE inference on the MAIA band: SVM has the lowest "
            "latency but requires CPU-side feature extraction per window. "
            "The ONNX-exported LSTM/CNN-LSTM/Conformer can run in onnxruntime "
            "with comparable latency on edge hardware."
        )
    if fastest_deep[0] == "Conformer":
        print(
            "\nConformer is the fastest deep model at inference, benefiting from "
            "fully parallel attention and depthwise convolution (no sequential "
            "hidden-state updates as in LSTM)."
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LSTM, CNN-LSTM, SVM, and Conformer on the EMG-ASL dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_models.py
  python scripts/benchmark_models.py --data-dir data/raw/synthetic/
  python scripts/benchmark_models.py --data-dir data/raw/ --epochs 30
        """,
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/synthetic/",
        dest="data_dir",
        help=(
            "Directory containing session CSV files. "
            "Synthetic data is generated here if the directory is missing or empty "
            "(default: data/raw/synthetic/)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for deep models (default: 30).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1: Ensure data exists
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/6  Preparing dataset")
    print("=" * 60)

    data_path = Path(args.data_dir)
    existing_csvs = list(data_path.glob("*.csv")) if data_path.exists() else []

    if not existing_csvs:
        print(
            f"  No CSV files found in '{data_path}'. "
            "Generating synthetic data (3 participants, 10 reps) …"
        )
        generate_dataset(
            output_dir=str(data_path),
            n_participants=3,
            n_reps=10,
            labels=list(ASL_LABELS),
        )
    else:
        print(f"  Found {len(existing_csvs)} CSV file(s) in '{data_path}'.")

    # ------------------------------------------------------------------
    # Step 2: Load & window
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2/6  Loading and windowing data")
    print("=" * 60)

    df = load_dataset(args.data_dir)
    windows, labels = create_windows(df)

    print(f"\n  Windows shape : {windows.shape}  (N, T, C)")
    print(f"  Unique labels : {len(np.unique(labels))}")

    # ------------------------------------------------------------------
    # Step 3: 80/20 split
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/6  Stratified 80/20 split")
    print("=" * 60)

    train_w, train_l, val_w, val_l = _stratified_split(windows, labels, val_fraction=0.20)
    print(f"  Train : {len(train_w)} windows")
    print(f"  Val   : {len(val_w)} windows")

    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Step 4a: LSTM
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 4a/6  LSTM  (epochs={args.epochs})")
    print("=" * 60)
    results["LSTM"] = _bench_lstm(train_w, train_l, val_w, val_l, epochs=args.epochs)
    print(
        f"  LSTM done — train_acc={results['LSTM']['train_acc']:.4f}  "
        f"val_acc={results['LSTM']['val_acc']:.4f}  "
        f"time={results['LSTM']['train_time']:.1f}s"
    )

    # ------------------------------------------------------------------
    # Step 4b: CNN-LSTM
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 4b/6  CNN-LSTM  (epochs={args.epochs})")
    print("=" * 60)
    results["CNN-LSTM"] = _bench_cnn_lstm(train_w, train_l, val_w, val_l, epochs=args.epochs)
    print(
        f"  CNN-LSTM done — train_acc={results['CNN-LSTM']['train_acc']:.4f}  "
        f"val_acc={results['CNN-LSTM']['val_acc']:.4f}  "
        f"time={results['CNN-LSTM']['train_time']:.1f}s"
    )

    # ------------------------------------------------------------------
    # Step 4c: SVM
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4c/6  SVM  (LinearSVC on 80-dim features)")
    print("=" * 60)
    results["SVM"] = _bench_svm(train_w, train_l, val_w, val_l)
    print(
        f"  SVM done — train_acc={results['SVM']['train_acc']:.4f}  "
        f"val_acc={results['SVM']['val_acc']:.4f}  "
        f"time={results['SVM']['train_time']:.1f}s"
    )

    # ------------------------------------------------------------------
    # Step 4d: Conformer
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 4d/6  Conformer  (epochs={args.epochs})")
    print("=" * 60)
    results["Conformer"] = _bench_conformer(train_w, train_l, val_w, val_l, epochs=args.epochs)
    print(
        f"  Conformer done — train_acc={results['Conformer']['train_acc']:.4f}  "
        f"val_acc={results['Conformer']['val_acc']:.4f}  "
        f"time={results['Conformer']['train_time']:.1f}s"
    )

    # ------------------------------------------------------------------
    # Step 5: Results table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 5/6  Comparison table")
    print("=" * 60)

    _print_table(results)

    # ------------------------------------------------------------------
    # Step 6: Conclusions
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 6/6  Conclusions")
    print("=" * 60)
    _print_winner(results)


if __name__ == "__main__":
    main()

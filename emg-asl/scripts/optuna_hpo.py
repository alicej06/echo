"""
Hyperparameter optimisation for the EMG-ASL classifiers using Optuna.

Searches over the LSTM and CNN-LSTM classifier hyperparameter spaces using
TPE (Tree-structured Parzen Estimator) sampling.  Each trial re-creates the
sliding windows with a trial-specific overlap, performs a stratified 80/20
split, and trains for up to 30 epochs with early stopping (patience=5).
The objective to maximise is validation accuracy.

After all trials:
  - Top-5 configurations are printed to stdout.
  - The best configuration is saved as JSON to ``--output-dir/best_hparams.json``.
  - With ``--train-best``, a final model is trained for the full epoch budget
    (50 epochs) with the best hyperparameters and saved as both .pt and .onnx.

Usage
-----
    # LSTM search on synthetic data (default):
    python scripts/optuna_hpo.py --n-trials 50 --data-dir data/raw/synthetic/

    # CNN-LSTM search on all real data:
    python scripts/optuna_hpo.py --n-trials 100 --data-dir data/raw/ --model cnn_lstm

    # Search then train final model:
    python scripts/optuna_hpo.py --n-trials 20 --train-best

Search space
------------
    hidden_size    : categorical [64, 128, 256]          (LSTM only)
    num_layers     : categorical [1, 2, 3]               (LSTM only)
    dropout        : categorical [0.1, 0.2, 0.3, 0.4, 0.5]
    lr             : log-uniform [1e-4, 1e-2]
    weight_decay   : log-uniform [1e-5, 1e-3]
    batch_size     : categorical [32, 64, 128]
    window_overlap : categorical [0.25, 0.5, 0.75]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure src.* imports work when invoked from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as exc:
    raise ImportError(
        "Optuna is required for HPO. Install it with: pip install optuna"
    ) from exc

from src.data.loader import create_windows, load_dataset
from src.data.synthetic import generate_dataset
from src.models.cnn_lstm_classifier import CNNLSTMClassifier
from src.models.lstm_classifier import ASLEMGClassifier
from src.utils.constants import ASL_LABELS, FEATURE_VECTOR_SIZE, NUM_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYNTHETIC_DATA_DIR: str = "data/raw/synthetic"
_N_PARTICIPANTS_SYNTHETIC: int = 5
_N_REPS_SYNTHETIC: int = 15

# Optuna trial training budget
_TRIAL_EPOCHS: int = 30
_TRIAL_EARLY_STOP: int = 5

# Final model training budget (when --train-best is set)
_FINAL_EPOCHS: int = 50
_FINAL_EARLY_STOP: int = 10

_VAL_FRACTION: float = 0.20
_RANDOM_SEED: int = 42
_DEVICE: str = "cpu"


# ---------------------------------------------------------------------------
# Label encoding
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
# Stratified split
# ---------------------------------------------------------------------------


def _stratified_split(
    windows: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = _VAL_FRACTION,
    seed: int = _RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified 80/20 split preserving per-class proportions.

    Returns (train_windows, train_labels, val_windows, val_labels).
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
# DataLoader factory
# ---------------------------------------------------------------------------


def _make_dataloader(
    windows: np.ndarray,
    labels_int: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X = torch.from_numpy(windows.astype(np.float32))
    y = torch.from_numpy(labels_int)
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Training loop (shared by both model types, parameterised via callables)
# ---------------------------------------------------------------------------


def _run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    early_stop_patience: int,
    checkpoint_path: Path,
    trial: "optuna.Trial | None" = None,
) -> float:
    """
    Generic training loop used by both LSTM and CNN-LSTM objective functions.

    Parameters
    ----------
    model : nn.Module
        Freshly constructed, un-trained model.
    train_loader, val_loader : DataLoader
        Pre-built data loaders.
    lr, weight_decay : float
        Optimiser hyperparameters.
    max_epochs, early_stop_patience : int
        Training budget and early stopping window.
    checkpoint_path : Path
        Temporary checkpoint path for this trial.
    trial : optuna.Trial or None
        When provided, intermediate values are reported so Optuna can prune.

    Returns
    -------
    float
        Best validation accuracy seen during training (value to maximise).
    """
    device = torch.device(_DEVICE)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        # ---- Train ----
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0
            model.save(checkpoint_path)  # type: ignore[attr-defined]
        else:
            no_improve += 1

        # Optuna intermediate reporting + pruning
        if trial is not None:
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if no_improve >= early_stop_patience:
            break

    return best_val_acc


# ---------------------------------------------------------------------------
# LSTM objective
# ---------------------------------------------------------------------------


def _lstm_objective(
    trial: "optuna.Trial",
    df_cache: dict,
    data_dir: str,
) -> float:
    """
    Optuna objective function for the LSTM classifier.

    The global DataFrame is cached in *df_cache* after the first load to
    avoid repeated disk I/O across trials.  Windowing is re-done every trial
    because window_overlap is a hyperparameter.

    Parameters
    ----------
    trial : optuna.Trial
    df_cache : dict
        Mutable dict used as a simple cache: keys ``"df"`` → DataFrame.
    data_dir : str
        Directory of session CSVs.

    Returns
    -------
    float
        Validation accuracy (to maximise).
    """
    # ---- Suggest hyperparameters ----
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    window_overlap = trial.suggest_categorical(
        "window_overlap", [0.25, 0.5, 0.75]
    )

    # ---- Load + window ----
    if "df" not in df_cache:
        df_cache["df"] = load_dataset(data_dir)
    df = df_cache["df"]

    windows, labels = create_windows(df, overlap=window_overlap)
    # LSTM uses flattened windows: (N, T*C)
    N, T, C = windows.shape
    windows_flat = windows.reshape(N, T * C)
    input_size = T * C

    train_win, train_lbl, val_win, val_lbl = _stratified_split(
        windows_flat, labels
    )

    train_y = _encode_labels(train_lbl, list(ASL_LABELS))
    val_y = _encode_labels(val_lbl, list(ASL_LABELS))

    train_loader = _make_dataloader(train_win, train_y, batch_size, shuffle=True)
    val_loader = _make_dataloader(val_win, val_y, batch_size, shuffle=False)

    # Single-layer LSTM must have dropout=0 between layers (PyTorch requirement)
    lstm_dropout = dropout if num_layers > 1 else 0.0
    model = ASLEMGClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        label_names=list(ASL_LABELS),
    )

    checkpoint_path = Path(f"models/optuna_trial_{trial.number}_lstm.pt")

    val_acc = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=_TRIAL_EPOCHS,
        early_stop_patience=_TRIAL_EARLY_STOP,
        checkpoint_path=checkpoint_path,
        trial=trial,
    )

    # Clean up temporary checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return val_acc


# ---------------------------------------------------------------------------
# CNN-LSTM objective
# ---------------------------------------------------------------------------


def _cnn_lstm_objective(
    trial: "optuna.Trial",
    df_cache: dict,
    data_dir: str,
) -> float:
    """
    Optuna objective function for the CNN-LSTM classifier.

    CNN-LSTM receives raw windows of shape (N, T, C) — no flattening.
    The architectural dimensions (CNN channels, LSTM hidden/layers) are fixed
    in the class definition; only dropout, optimiser settings, batch size, and
    window overlap are searched.

    Parameters
    ----------
    trial : optuna.Trial
    df_cache : dict
        Mutable dict used as a simple cache.
    data_dir : str
        Directory of session CSVs.

    Returns
    -------
    float
        Validation accuracy (to maximise).
    """
    # ---- Suggest hyperparameters ----
    # hidden_size and num_layers are fixed in CNNLSTMClassifier architecture,
    # so we still suggest them for logging / future extensibility but they are
    # not passed to the constructor.
    trial.suggest_categorical("hidden_size", [64, 128, 256])   # informational
    trial.suggest_categorical("num_layers", [1, 2, 3])          # informational
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    window_overlap = trial.suggest_categorical(
        "window_overlap", [0.25, 0.5, 0.75]
    )

    # ---- Load + window ----
    if "df" not in df_cache:
        df_cache["df"] = load_dataset(data_dir)
    df = df_cache["df"]

    windows, labels = create_windows(df, overlap=window_overlap)
    # CNN-LSTM consumes raw (N, T, C) — no flatten.

    train_win, train_lbl, val_win, val_lbl = _stratified_split(windows, labels)

    train_y = _encode_labels(train_lbl, list(ASL_LABELS))
    val_y = _encode_labels(val_lbl, list(ASL_LABELS))

    train_loader = _make_dataloader(train_win, train_y, batch_size, shuffle=True)
    val_loader = _make_dataloader(val_win, val_y, batch_size, shuffle=False)

    model = CNNLSTMClassifier(
        num_classes=NUM_CLASSES,
        dropout=dropout,
        label_names=list(ASL_LABELS),
    )

    checkpoint_path = Path(f"models/optuna_trial_{trial.number}_cnn_lstm.pt")

    val_acc = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=_TRIAL_EPOCHS,
        early_stop_patience=_TRIAL_EARLY_STOP,
        checkpoint_path=checkpoint_path,
        trial=trial,
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return val_acc


# ---------------------------------------------------------------------------
# Final model training with best hyperparameters
# ---------------------------------------------------------------------------


def _train_final_lstm(
    best_params: dict[str, Any],
    data_dir: str,
    output_dir: Path,
) -> None:
    """Train a final LSTM with best_params for _FINAL_EPOCHS epochs."""
    print("\n[train_best] Training final LSTM with best hyperparameters...")

    df = load_dataset(data_dir)
    overlap = best_params["window_overlap"]
    windows, labels = create_windows(df, overlap=overlap)

    N, T, C = windows.shape
    windows_flat = windows.reshape(N, T * C)
    input_size = T * C

    train_win, train_lbl, val_win, val_lbl = _stratified_split(
        windows_flat, labels
    )
    train_y = _encode_labels(train_lbl, list(ASL_LABELS))
    val_y = _encode_labels(val_lbl, list(ASL_LABELS))

    batch_size = best_params["batch_size"]
    train_loader = _make_dataloader(train_win, train_y, batch_size, shuffle=True)
    val_loader = _make_dataloader(val_win, val_y, batch_size, shuffle=False)

    model = ASLEMGClassifier(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        num_classes=NUM_CLASSES,
        dropout=best_params["dropout"],
        label_names=list(ASL_LABELS),
    )

    checkpoint_path = output_dir / "best_lstm_final.pt"
    val_acc = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        max_epochs=_FINAL_EPOCHS,
        early_stop_patience=_FINAL_EARLY_STOP,
        checkpoint_path=checkpoint_path,
    )

    # Reload best checkpoint and export
    final_model = ASLEMGClassifier.load(checkpoint_path)

    pt_out = output_dir / "best_lstm_classifier.pt"
    onnx_out = output_dir / "best_lstm_classifier.onnx"
    final_model.save(pt_out)
    final_model.to_onnx(onnx_out)

    print(f"[train_best] Final val accuracy : {val_acc:.4f}  ({val_acc * 100:.2f} %)")
    print(f"[train_best] PyTorch model      : {pt_out}")
    print(f"[train_best] ONNX model         : {onnx_out}")

    # Clean up intermediate checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def _train_final_cnn_lstm(
    best_params: dict[str, Any],
    data_dir: str,
    output_dir: Path,
) -> None:
    """Train a final CNN-LSTM with best_params for _FINAL_EPOCHS epochs."""
    print("\n[train_best] Training final CNN-LSTM with best hyperparameters...")

    df = load_dataset(data_dir)
    overlap = best_params["window_overlap"]
    windows, labels = create_windows(df, overlap=overlap)

    train_win, train_lbl, val_win, val_lbl = _stratified_split(windows, labels)
    train_y = _encode_labels(train_lbl, list(ASL_LABELS))
    val_y = _encode_labels(val_lbl, list(ASL_LABELS))

    batch_size = best_params["batch_size"]
    train_loader = _make_dataloader(train_win, train_y, batch_size, shuffle=True)
    val_loader = _make_dataloader(val_win, val_y, batch_size, shuffle=False)

    model = CNNLSTMClassifier(
        num_classes=NUM_CLASSES,
        dropout=best_params["dropout"],
        label_names=list(ASL_LABELS),
    )

    checkpoint_path = output_dir / "best_cnn_lstm_final.pt"
    val_acc = _run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        max_epochs=_FINAL_EPOCHS,
        early_stop_patience=_FINAL_EARLY_STOP,
        checkpoint_path=checkpoint_path,
    )

    final_model = CNNLSTMClassifier.load(checkpoint_path)

    pt_out = output_dir / "best_cnn_lstm_classifier.pt"
    onnx_out = output_dir / "best_cnn_lstm_classifier.onnx"
    final_model.save(pt_out)
    final_model.to_onnx(onnx_out)

    print(f"[train_best] Final val accuracy : {val_acc:.4f}  ({val_acc * 100:.2f} %)")
    print(f"[train_best] PyTorch model      : {pt_out}")
    print(f"[train_best] ONNX model         : {onnx_out}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for EMG-ASL classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=_SYNTHETIC_DATA_DIR,
        help=(
            "Directory containing session CSV files.  "
            "If data/raw/synthetic/ and empty, synthetic data is generated first."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "cnn_lstm"],
        default="lstm",
        help="Which model architecture to optimise.",
    )
    parser.add_argument(
        "--train-best",
        action="store_true",
        default=False,
        help=(
            "After the search, train a final model for the full epoch budget "
            "using the best hyperparameters found."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory where best_hparams.json and final models are written.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir

    # ------------------------------------------------------------------
    # Auto-generate synthetic data if the target directory is empty and
    # matches the default synthetic path.
    # ------------------------------------------------------------------
    data_path = Path(data_dir)
    is_synthetic_dir = (
        data_path.resolve()
        == (Path(_PROJECT_ROOT) / _SYNTHETIC_DATA_DIR).resolve()
    ) or str(data_dir).endswith("synthetic")

    existing_csvs = list(data_path.glob("*.csv")) if data_path.exists() else []
    if is_synthetic_dir and len(existing_csvs) < _N_PARTICIPANTS_SYNTHETIC:
        print(
            f"[optuna_hpo] No CSV files found in '{data_path}'. "
            "Generating synthetic data first..."
        )
        generate_dataset(
            output_dir=str(data_path),
            n_participants=_N_PARTICIPANTS_SYNTHETIC,
            n_reps=_N_REPS_SYNTHETIC,
            labels=list(ASL_LABELS),
        )

    # ------------------------------------------------------------------
    # Build Optuna study
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Optuna HPO  |  model={args.model}  |  n_trials={args.n_trials}")
    print(f"data_dir    : {data_dir}")
    print(f"output_dir  : {output_dir}")
    print("=" * 60)
    print()

    # Suppress per-trial Optuna INFO messages for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=_RANDOM_SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"emg_asl_{args.model}_hpo",
    )

    # Shared DataFrame cache — avoids reloading disk on every trial.
    df_cache: dict = {}

    if args.model == "lstm":
        objective_fn = lambda trial: _lstm_objective(trial, df_cache, data_dir)
    else:
        objective_fn = lambda trial: _cnn_lstm_objective(
            trial, df_cache, data_dir
        )

    t_start = time.time()

    study.optimize(
        objective_fn,
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(Exception,),  # continue study even if a trial crashes
    )

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        print("[optuna_hpo] ERROR: No trials completed successfully. Exiting.")
        sys.exit(1)

    # Sort by objective (descending)
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)

    print()
    print("=" * 60)
    print(f"HPO complete  |  {len(completed)} / {args.n_trials} trials finished")
    print(f"Total time    : {elapsed:.1f} s")
    print()
    print("Top-5 configurations:")
    print("-" * 60)

    top5 = sorted_trials[:5]
    for rank, trial in enumerate(top5, start=1):
        print(f"  #{rank}  val_acc={trial.value:.4f}  params={trial.params}")

    print("-" * 60)

    best_trial = sorted_trials[0]
    best_params = best_trial.params

    print(f"\nBest trial   : #{best_trial.number}")
    print(f"Best val_acc : {best_trial.value:.4f}  ({best_trial.value * 100:.2f} %)")
    print(f"Best params  : {json.dumps(best_params, indent=2)}")

    # ------------------------------------------------------------------
    # Save best hyperparameters to JSON
    # ------------------------------------------------------------------
    hparams_path = output_dir / "best_hparams.json"
    with open(hparams_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "val_accuracy": best_trial.value,
                "trial_number": best_trial.number,
                "hyperparameters": best_params,
            },
            f,
            indent=2,
        )
    print(f"\nBest hyperparameters saved to {hparams_path}")

    # ------------------------------------------------------------------
    # Optionally train a final model with the best config
    # ------------------------------------------------------------------
    if args.train_best:
        if args.model == "lstm":
            _train_final_lstm(best_params, data_dir, output_dir)
        else:
            _train_final_cnn_lstm(best_params, data_dir, output_dir)

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

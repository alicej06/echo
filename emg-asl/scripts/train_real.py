"""
Train the EMG-ASL classifier on real participant data.

Loads all session CSV files from --data-dir, creates sliding windows,
then either performs leave-one-participant-out cross-validation (--lopo)
or an 80/10/10 stratified train/val/test split.

The best model is exported as both a PyTorch checkpoint and an ONNX graph.

Usage (from project root)
--------------------------
    python scripts/train_real.py --data-dir data/raw/ --epochs 100 --output models/
    python scripts/train_real.py --data-dir data/raw/ --lopo
    python scripts/train_real.py --data-dir data/raw/ --epochs 50 --wandb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.*`` imports work when
# invoked as ``python scripts/train_real.py`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.data.loader import create_windows, load_dataset
from src.models.train import TrainConfig, evaluate_model, lopo_cross_validate, train_model
from src.utils.constants import ASL_LABELS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def _stratified_split_80_10_10(
    windows: np.ndarray,
    labels: np.ndarray,
    participant_ids: np.ndarray,
    seed: int = 42,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Stratified 80/10/10 split by participant.

    Participants are randomly assigned so that approximately 80 % of
    participants form the training set, 10 % validation, and 10 % test.
    Within each split all windows for that participant are kept together
    (no within-participant leakage across splits).

    Falls back to a per-window stratified split when there are fewer than
    3 unique participants.
    """
    rng = np.random.default_rng(seed)
    unique_participants = np.array(sorted(set(participant_ids)))

    if len(unique_participants) >= 3:
        rng.shuffle(unique_participants)
        n = len(unique_participants)
        n_val = max(1, round(n * 0.10))
        n_test = max(1, round(n * 0.10))
        n_train = n - n_val - n_test

        train_parts = set(unique_participants[:n_train])
        val_parts = set(unique_participants[n_train: n_train + n_val])
        test_parts = set(unique_participants[n_train + n_val:])

        train_mask = np.array([p in train_parts for p in participant_ids])
        val_mask = np.array([p in val_parts for p in participant_ids])
        test_mask = np.array([p in test_parts for p in participant_ids])

        print(
            f"[split] Participants -> train: {sorted(train_parts)}, "
            f"val: {sorted(val_parts)}, test: {sorted(test_parts)}"
        )
    else:
        # Fall back to per-window stratified split
        print(
            "[split] Fewer than 3 participants — using per-window stratified split."
        )
        unique_labels = np.unique(labels)
        train_idxs, val_idxs, test_idxs = [], [], []
        for lbl in unique_labels:
            idxs = np.where(labels == lbl)[0]
            rng.shuffle(idxs)
            n = len(idxs)
            n_val = max(1, round(n * 0.10))
            n_test = max(1, round(n * 0.10))
            val_idxs.extend(idxs[:n_val].tolist())
            test_idxs.extend(idxs[n_val: n_val + n_test].tolist())
            train_idxs.extend(idxs[n_val + n_test:].tolist())

        train_mask = np.zeros(len(labels), dtype=bool)
        val_mask = np.zeros(len(labels), dtype=bool)
        test_mask = np.zeros(len(labels), dtype=bool)
        train_mask[train_idxs] = True
        val_mask[val_idxs] = True
        test_mask[test_idxs] = True

    return (
        windows[train_mask], labels[train_mask],
        windows[val_mask],   labels[val_mask],
        windows[test_mask],  labels[test_mask],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the EMG-ASL classifier on real participant data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_real.py --data-dir data/raw/ --epochs 100 --output models/
  python scripts/train_real.py --data-dir data/raw/ --lopo
  python scripts/train_real.py --data-dir data/raw/ --epochs 50 --wandb
        """,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        dest="data_dir",
        help="Directory containing session CSV files (output of record_session.py).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--output",
        default="models/",
        help="Output directory for saved models (default: models/).",
    )
    parser.add_argument(
        "--lopo",
        action="store_true",
        help="Run leave-one-participant-out cross-validation instead of a fixed split.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        dest="hidden_size",
        help="LSTM hidden size (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="batch_size",
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    args = parser.parse_args()

    t_total = time.time()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / "asl_emg_classifier.pt"
    onnx_path = output_dir / "asl_emg_classifier.onnx"

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/4  Loading dataset")
    print("=" * 60)
    df = load_dataset(args.data_dir)

    if "participant_id" not in df.columns:
        raise RuntimeError(
            "Combined DataFrame is missing 'participant_id' column. "
            "Ensure CSV files follow the naming convention: "
            "{participant_id}_{date}.csv"
        )

    # ------------------------------------------------------------------
    # Step 2: Windowing
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 2/4  Creating sliding windows  (N, 40, 8)")
    print("=" * 60)
    windows, labels = create_windows(df)

    # Build a matching participant_id array aligned with windows.
    # create_windows preserves original row order so we can index df.
    # Re-derive by repeating the per-row participant_id through the windowing.
    # Simpler: re-run the sliding window index tracking.
    from src.utils.constants import OVERLAP, SAMPLE_RATE, WINDOW_SIZE_MS

    _CHANNEL_COLS = [f"ch{i + 1}" for i in range(8)]
    window_samples = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
    step_samples = max(1, int(window_samples * (1.0 - OVERLAP)))
    label_arr = df["label"].values
    pid_arr = df["participant_id"].values

    window_pids = []
    for start in range(0, len(label_arr) - window_samples + 1, step_samples):
        end = start + window_samples
        window_labels_slice = label_arr[start:end]
        unique_l = set(window_labels_slice)
        if len(unique_l) != 1:
            continue
        lbl = next(iter(unique_l))
        if not lbl or lbl == "nan":
            continue
        window_pids.append(pid_arr[start])

    participant_ids = np.array(window_pids)
    assert len(participant_ids) == len(windows), (
        f"Participant ID array length mismatch: {len(participant_ids)} vs {len(windows)}"
    )

    print(f"  Windows shape     : {windows.shape}  (N, T, C)")
    print(f"  Unique labels     : {len(np.unique(labels))}")
    print(f"  Unique participants: {sorted(set(participant_ids))}")

    # Base training config (wandb_project empty = disabled unless --wandb)
    base_config = TrainConfig(
        hidden_size=args.hidden_size,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        lr=args.lr,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=10,
        lr_patience=5,
        lr_factor=0.5,
        lr_min=1e-6,
        checkpoint_dir=str(output_dir),
        checkpoint_name="best_model.pt",
        device="cpu",
        wandb_project="emg-asl" if args.wandb else "",
        wandb_run_name="real-data",
        label_names=list(ASL_LABELS),
    )

    # ------------------------------------------------------------------
    # Step 3: Train
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    if args.lopo:
        print("Step 3/4  Leave-one-participant-out cross-validation")
    else:
        print("Step 3/4  Training  (80/10/10 stratified split)")
    print("=" * 60)

    if args.lopo:
        results = lopo_cross_validate(
            all_windows=windows,
            all_labels=labels,
            participant_ids=participant_ids,
            config=base_config,
        )
        print(f"\n[LOPO CV] Mean accuracy : {results['mean_acc']:.4f}")
        print(f"[LOPO CV] Std accuracy  : {results['std_acc']:.4f}")
        for fold in results["per_fold"]:
            print(f"  Fold {fold['fold_id']:>6s} : {fold['accuracy']:.4f}")

        elapsed = time.time() - t_total
        print(f"\nTotal time: {elapsed:.1f} s")
        print("Note: in LOPO mode the final model is the last fold's model.")
        print(f"  PyTorch checkpoint : {output_dir / 'best_model.pt'}")
        # Still export the last checkpoint to canonical paths
        import shutil
        last_pt = output_dir / f"best_model_fold{len(results['per_fold']) - 1}.pt"
        if last_pt.exists():
            shutil.copy(last_pt, pt_path)
            print(f"  Copied last fold checkpoint -> {pt_path}")
        return

    # Fixed split path
    print()
    print("=" * 60)
    print("Step 3a/4  Splitting dataset  (80 / 10 / 10, stratified by participant)")
    print("=" * 60)

    (
        train_w, train_l,
        val_w, val_l,
        test_w, test_l,
    ) = _stratified_split_80_10_10(windows, labels, participant_ids)

    print(f"  Train : {len(train_w)} windows")
    print(f"  Val   : {len(val_w)} windows")
    print(f"  Test  : {len(test_w)} windows")

    print()
    print("=" * 60)
    print("Step 3b/4  Training ASLEMGClassifier")
    print("=" * 60)
    print(
        f"  Config: epochs={base_config.epochs}, "
        f"batch_size={base_config.batch_size}, "
        f"hidden_size={base_config.hidden_size}, "
        f"lr={base_config.lr}"
    )
    print()

    model = train_model(
        train_windows=train_w,
        train_labels=train_l,
        val_windows=val_w,
        val_labels=val_l,
        config=base_config,
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluate on test set
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 4/4  Evaluating on held-out test set")
    print("=" * 60)

    metrics = evaluate_model(
        model=model,
        test_windows=test_w,
        test_labels=test_l,
        label_names=list(ASL_LABELS),
    )
    test_acc = metrics["accuracy"]

    # ------------------------------------------------------------------
    # Save PyTorch checkpoint
    # ------------------------------------------------------------------
    model.save(pt_path)
    print(f"\nPyTorch model saved to {pt_path}")

    # ------------------------------------------------------------------
    # Export ONNX
    # ------------------------------------------------------------------
    model.to_onnx(onnx_path)
    print(f"ONNX model exported to {onnx_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_total
    print()
    print("=" * 60)
    print(f"Final test accuracy : {test_acc:.4f}  ({test_acc * 100:.2f} %)")
    print(f"Total time          : {elapsed:.1f} s")
    print(f"PyTorch checkpoint  : {pt_path}")
    print(f"ONNX model          : {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

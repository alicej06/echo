"""
Training script for the CrossModalASL dual-encoder embedding model.

What this script does
----------------------
1. Optionally generates synthetic EMG data if the emg-dir is empty.
2. Calls train_cross_modal() to train the dual-encoder via InfoNCE loss.
3. After training, loads 20 random EMG windows from the dataset and
   classifies them via the vision gallery.  Prints per-class hits and
   overall accuracy as a quick sanity check.
4. Saves the trained model to {output-dir}/cross_modal_asl.pt and the
   class gallery to {output-dir}/class_gallery.npy.

Self-supervised mode (no --video-dir)
--------------------------------------
When --video-dir is not provided, the script trains using synthetic
landmark stand-ins (random unit-vector prototypes per class with small
noise).  This is useful for integration testing and rapid iteration
but the resulting embeddings do NOT align with real hand appearance.

Usage (from project root)
--------------------------
    # Self-supervised mode (no real video):
    python scripts/train_cross_modal.py --emg-dir data/raw/synthetic

    # With real paired video (when available):
    python scripts/train_cross_modal.py \\
        --emg-dir data/raw \\
        --video-dir data/video \\
        --epochs 100 \\
        --output-dir models/cross_modal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to sys.path so that ``from src.*`` imports resolve when
# this script is executed directly from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CrossModalASL dual-encoder embedding model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--emg-dir",
        type=str,
        required=True,
        help=(
            "Directory containing labeled EMG session CSVs "
            "(or where synthetic data will be generated if empty)."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing paired video files. "
            "If omitted, training runs in self-supervised mode with "
            "synthetic vision embeddings."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for contrastive training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Directory where the trained model and gallery will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Number of random EMG windows to classify during post-training eval.",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help=(
            "Generate synthetic EMG data into --emg-dir before training "
            "(if the directory is empty or does not exist)."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Synthetic data generation helper
# ---------------------------------------------------------------------------


def _maybe_generate_synthetic(emg_dir: str, seed: int) -> None:
    """Generate synthetic EMG CSVs into emg_dir if it is empty."""
    from src.data.synthetic import generate_dataset

    out_path = Path(emg_dir)
    existing = list(out_path.glob("*.csv")) if out_path.exists() else []

    if existing:
        print(
            f"  Found {len(existing)} existing CSV(s) in '{emg_dir}'. "
            "Skipping synthetic generation."
        )
        return

    print(
        f"  No CSVs found in '{emg_dir}'. "
        "Generating 3 synthetic participants ..."
    )
    generate_dataset(
        output_dir=emg_dir,
        n_participants=3,
        n_reps=10,
    )


# ---------------------------------------------------------------------------
# Quick post-training evaluation
# ---------------------------------------------------------------------------


def _quick_eval(
    model_path: str,
    gallery_path: str,
    emg_dir: str,
    n_samples: int,
    seed: int,
) -> float:
    """Classify n_samples random EMG windows and return accuracy.

    Loads the trained model and gallery from disk to verify the full
    save/load round-trip as well as the classification pipeline.

    Parameters
    ----------
    model_path:
        Path to the saved CrossModalASL .pt checkpoint.
    gallery_path:
        Path to the saved class_gallery.npy file.
    emg_dir:
        EMG CSV directory (same one used for training).
    n_samples:
        How many windows to sample for the evaluation.
    seed:
        Random seed.

    Returns
    -------
    float
        Fraction of correctly classified samples in [0, 1].
    """
    from src.data.loader import create_windows, load_dataset
    from src.models.cross_modal_embedding import CrossModalASL

    rng = np.random.default_rng(seed + 1)

    # Load model and gallery.
    model = CrossModalASL.load(model_path)
    gallery: dict[str, np.ndarray] = np.load(  # type: ignore[assignment]
        gallery_path, allow_pickle=True
    ).item()

    # Load EMG windows.
    df = load_dataset(emg_dir)
    from src.utils.constants import ASL_LABELS
    known = set(ASL_LABELS)
    df = df[df["label"].isin(known)].reset_index(drop=True)

    windows, str_labels = create_windows(df)
    N, T, C = windows.shape
    emg_flat = windows.reshape(N, T * C).astype(np.float32)

    # Sample up to n_samples windows.
    n_eval = min(n_samples, N)
    eval_idx = rng.choice(N, size=n_eval, replace=False)

    print(f"\nPost-training evaluation on {n_eval} random windows ...")
    print(f"  {'True':>12s}  {'Predicted':>12s}  {'Score':>8s}  Match")
    print("  " + "-" * 50)

    correct = 0
    for i in eval_idx:
        true_lbl = str_labels[i]
        pred_lbl, score = model.classify_emg(emg_flat[i], gallery)
        match = pred_lbl == true_lbl
        if match:
            correct += 1
        print(
            f"  {true_lbl:>12s}  {pred_lbl:>12s}  {score:>8.4f}  "
            + ("OK" if match else "--")
        )

    acc = correct / n_eval if n_eval > 0 else 0.0
    print(f"\n  Accuracy: {correct}/{n_eval} = {acc:.4f} ({acc * 100:.1f} %)")
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    t_start = time.time()

    # Print mode banner.
    print("=" * 64)
    print("CrossModalASL -- training script")
    print("=" * 64)

    if args.video_dir is None:
        print(
            "\nNo video dir provided. Training in self-supervised mode with "
            "synthetic vision embeddings. Provide --video-dir for real "
            "cross-modal training.\n"
        )
    else:
        print(f"\nVideo directory : {args.video_dir}")

    print(f"EMG directory   : {args.emg_dir}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Seed            : {args.seed}")
    print()

    # Optionally generate synthetic data.
    if args.generate_synthetic:
        print("Step 0 -- Generating synthetic EMG data ...")
        _maybe_generate_synthetic(args.emg_dir, args.seed)
        print()

    # Verify the EMG directory contains CSVs before proceeding.
    emg_path = Path(args.emg_dir)
    csv_files = list(emg_path.glob("*.csv")) if emg_path.exists() else []
    if not csv_files:
        print(
            f"[ERROR] No CSV files found in '{args.emg_dir}'.\n"
            "        Re-run with --generate-synthetic to create synthetic data,\n"
            "        or point --emg-dir to a directory containing session CSVs."
        )
        sys.exit(1)

    # Import and run training.
    from src.models.cross_modal_embedding import train_cross_modal

    print("=" * 64)
    print("Training")
    print("=" * 64)

    model = train_cross_modal(
        emg_csv_dir=args.emg_dir,
        video_dir=args.video_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Confirm artifact paths.
    out_path = Path(args.output_dir)
    model_save_path = str(out_path / "cross_modal_asl.pt")
    gallery_save_path = str(out_path / "class_gallery.npy")

    # Post-training quick evaluation.
    print()
    print("=" * 64)
    print("Quick evaluation")
    print("=" * 64)

    acc = _quick_eval(
        model_path=model_save_path,
        gallery_path=gallery_save_path,
        emg_dir=args.emg_dir,
        n_samples=args.eval_samples,
        seed=args.seed,
    )

    # Summary.
    elapsed = time.time() - t_start
    print()
    print("=" * 64)
    print("Done")
    print("=" * 64)
    print(f"  Model checkpoint : {model_save_path}")
    print(f"  Class gallery    : {gallery_save_path}")
    print(f"  Quick eval acc   : {acc * 100:.1f} %")
    print(f"  Total time       : {elapsed:.1f} s")


if __name__ == "__main__":
    main()

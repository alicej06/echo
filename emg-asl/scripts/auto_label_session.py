"""
Auto-label an EMG session CSV using vision-derived ASL pose labels.

Takes a pre-recorded video alongside its corresponding EMG session CSV,
extracts per-frame ASL letter predictions using MediaPipe, syncs those labels
to EMG timestamps via nearest-neighbor matching, and writes the result to a
new CSV ready for use with train_real.py.

Also supports a ``--webcam`` mode that opens the camera live and streams
detected poses to stdout -- useful for testing that MediaPipe is running and
the pose classifier (once trained) produces sensible output without needing
any EMG data.

Usage examples
--------------
Offline labeling (most common):

    python scripts/auto_label_session.py \\
        --video  data/raw/P001_20260227.mp4 \\
        --emg    data/raw/P001_20260227_145301.csv \\
        --output data/processed/P001_20260227_autolabeled.csv

Use default output path (same directory as EMG, stem + '_autolabeled.csv'):

    python scripts/auto_label_session.py \\
        --video data/raw/P001_20260227.mp4 \\
        --emg   data/raw/P001_20260227_145301.csv

Live webcam pose test (no EMG required):

    python scripts/auto_label_session.py --webcam

Load a pre-trained pose classifier (optional):

    python scripts/auto_label_session.py \\
        --video     data/raw/P001_20260227.mp4 \\
        --emg       data/raw/P001_20260227_145301.csv \\
        --classifier models/pose_classifier.joblib

Adjust confidence and sync tolerance:

    python scripts/auto_label_session.py \\
        --video      data/raw/P001.mp4 \\
        --emg        data/raw/P001.csv \\
        --confidence 0.80 \\
        --tolerance  30
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Make ``from src.*`` imports work when this script is run directly from
# the project root as ``python scripts/auto_label_session.py``.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data.vision_teacher import (
    VisionTeacher,
    load_trained_classifier,
    sync_labels_to_emg,
)
from src.utils.constants import CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _print_label_distribution(series: pd.Series, top_n: int = 10) -> None:
    """Print label counts, sorted by frequency descending."""
    counts = Counter(series)
    total = len(series)
    print(f"  {'Label':<15} {'Count':>7}  {'%':>6}")
    print(f"  {'-' * 32}")
    for label, count in counts.most_common(top_n):
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {label:<15} {count:>7}  {pct:>5.1f}%")
    if len(counts) > top_n:
        remaining = sum(v for _, v in counts.most_common()[top_n:])
        print(f"  ... ({len(counts) - top_n} more labels, {remaining} rows)")


# ---------------------------------------------------------------------------
# Offline labeling mode
# ---------------------------------------------------------------------------


def run_offline(args: argparse.Namespace) -> None:
    """Label a video + EMG pair and save the merged CSV."""

    video_path = Path(args.video)
    emg_path = Path(args.emg)

    # Validate inputs.
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    if not emg_path.exists():
        print(f"[ERROR] EMG CSV not found: {emg_path}")
        sys.exit(1)

    # Resolve output path.
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = emg_path.parent / (emg_path.stem + "_autolabeled.csv")

    _print_section("EMG-ASL Auto-Labeler")
    print(f"  Video       : {video_path}")
    print(f"  EMG CSV     : {emg_path}")
    print(f"  Output      : {out_path}")
    print(f"  Confidence  : >= {args.confidence:.2f}")
    print(f"  Sync tolerance : {args.tolerance} ms")
    if args.classifier:
        print(f"  Classifier  : {args.classifier}")
    else:
        print(
            "  Classifier  : UNTRAINED baseline (all predictions will be UNKNOWN)"
        )

    # Load EMG data.
    print("\nLoading EMG CSV ... ", end="", flush=True)
    try:
        emg_df = pd.read_csv(emg_path)
    except Exception as exc:
        print(f"\n[ERROR] Could not read EMG CSV: {exc}")
        sys.exit(1)
    print(f"{len(emg_df):,} rows.")

    # Build VisionTeacher.
    teacher = VisionTeacher(confidence_threshold=0.0)  # keep all; filter later

    if args.classifier:
        clf_path = Path(args.classifier)
        if not clf_path.exists():
            print(f"[ERROR] Classifier file not found: {clf_path}")
            sys.exit(1)
        print(f"Loading pose classifier from {clf_path} ... ", end="", flush=True)
        try:
            teacher._classifier = load_trained_classifier(str(clf_path))
            print("OK.")
        except Exception as exc:
            print(f"\n[ERROR] Failed to load classifier: {exc}")
            sys.exit(1)

    # Extract labels from the video.
    _print_section("Step 1 / 2 -- Extracting vision labels from video")
    t0 = time.monotonic()
    print(f"Processing {video_path.name} ... (this may take a while)")

    try:
        label_df = teacher.label_video_file(str(video_path))
    except ImportError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f} s.")
    print(f"  Total frames with hand detected : {len(label_df):,}")

    # Apply confidence filter for the sync step.
    label_df_filtered = label_df[label_df["confidence"] >= args.confidence].copy()
    print(f"  Frames above confidence {args.confidence:.2f}       : {len(label_df_filtered):,}")

    if label_df_filtered.empty:
        print(
            "\n[WARNING] No vision labels passed the confidence threshold.\n"
            "          The output will have all rows labeled UNLABELED.\n"
            "          Consider lowering --confidence or loading a trained classifier."
        )

    # Sync labels to EMG timestamps.
    _print_section("Step 2 / 2 -- Syncing labels to EMG timestamps")
    print(f"Matching {len(emg_df):,} EMG rows to {len(label_df_filtered):,} vision labels ...")

    labeled_df = sync_labels_to_emg(
        label_df=label_df_filtered,
        emg_df=emg_df,
        tolerance_ms=args.tolerance,
    )

    # Compute stats.
    n_labeled = (labeled_df["label"] != "UNLABELED").sum()
    n_total = len(labeled_df)
    pct = 100.0 * n_labeled / n_total if n_total > 0 else 0.0

    print(f"\n  Total EMG rows   : {n_total:,}")
    print(f"  Labeled rows     : {n_labeled:,}  ({pct:.1f}%)")
    print(f"  Unlabeled rows   : {n_total - n_labeled:,}")

    # Label distribution (exclude UNLABELED for clarity).
    labeled_only = labeled_df[labeled_df["label"] != "UNLABELED"]["label"]
    if not labeled_only.empty:
        print("\n  Label distribution (labeled rows only):")
        _print_label_distribution(labeled_only)

    # Save.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out_path, index=False)

    _print_section("Done")
    print(f"  Saved to: {out_path.resolve()}")
    print()


# ---------------------------------------------------------------------------
# Webcam live test mode
# ---------------------------------------------------------------------------


def run_webcam(args: argparse.Namespace) -> None:
    """Open the default camera and stream detected poses to stdout."""

    _print_section("VisionTeacher -- Live Webcam Test")
    print("  Show an ASL hand pose to the camera.")
    print(f"  Minimum confidence to print: {args.confidence:.2f}")
    if args.classifier:
        print(f"  Classifier: {args.classifier}")
    else:
        print(
            "  Classifier: UNTRAINED baseline -- all predictions will be UNKNOWN.\n"
            "  Train the pose classifier and pass --classifier to see real letters."
        )
    print("  Press Ctrl+C to quit.\n")

    teacher = VisionTeacher(
        source=0, confidence_threshold=args.confidence
    )

    if args.classifier:
        clf_path = Path(args.classifier)
        if not clf_path.exists():
            print(f"[ERROR] Classifier not found: {clf_path}")
            sys.exit(1)
        try:
            teacher._classifier = load_trained_classifier(str(clf_path))
        except Exception as exc:
            print(f"[ERROR] Could not load classifier: {exc}")
            sys.exit(1)

    teacher.start_webcam(device_index=0)

    print(f"  {'timestamp_ms':>14}  {'label':<12}  confidence")
    print(f"  {'-' * 40}")

    try:
        for ts_ms, label, conf in teacher.stream_labels():
            print(f"  {ts_ms:>14.1f}  {label:<12}  {conf:.3f}")
    except KeyboardInterrupt:
        pass
    finally:
        teacher.stop()

    print("\n[Stopped]")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-label an EMG session using MediaPipe vision labels. "
            "Pass --webcam to run a live camera test with no EMG input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline: label a recorded session
  python scripts/auto_label_session.py --video data/raw/P001.mp4 --emg data/raw/P001.csv

  # Webcam live test
  python scripts/auto_label_session.py --webcam

  # With trained classifier and custom output path
  python scripts/auto_label_session.py \\
      --video data/raw/P001.mp4 \\
      --emg   data/raw/P001.csv \\
      --classifier models/pose_classifier.joblib \\
      --output data/processed/P001_labeled.csv
        """,
    )

    # Mode selection.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--webcam",
        action="store_true",
        help="Open the webcam and stream detected poses live (no EMG required).",
    )

    # Offline mode inputs.
    parser.add_argument(
        "--video",
        metavar="PATH",
        help="Path to the recorded video file (MP4, AVI, etc.).",
    )
    parser.add_argument(
        "--emg",
        metavar="PATH",
        help="Path to the EMG session CSV (columns: timestamp_ms, ch1..ch8).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Output CSV path. "
            "Default: <emg_stem>_autolabeled.csv in the same directory as --emg."
        ),
    )

    # Shared options.
    parser.add_argument(
        "--classifier",
        metavar="PATH",
        default=None,
        help=(
            "Path to a trained pose classifier .joblib file saved by "
            "SimpleASLPoseClassifier.save(). "
            "If omitted, the untrained baseline is used (all UNKNOWN)."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        metavar="FLOAT",
        help=(
            f"Minimum confidence [0-1] for a label to be accepted "
            f"(default: {CONFIDENCE_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=50.0,
        metavar="MS",
        help=(
            "Maximum timestamp gap in ms for EMG/vision matching "
            "(default: 50 ms = 10 samples at 200 Hz)."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.webcam:
        run_webcam(args)
    else:
        # Offline mode requires both --video and --emg.
        missing = []
        if not args.video:
            missing.append("--video")
        if not args.emg:
            missing.append("--emg")
        if missing:
            parser.error(
                f"Offline mode requires: {', '.join(missing)}.  "
                "Use --webcam for live-camera testing without an EMG file."
            )
        run_offline(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract MediaPipe hand landmarks from downloaded WLASL videos.

Run this script after scripts/download_wlasl.py to produce the
data/wlasl/landmarks.npz file consumed by scripts/train_gpu_ddp.py
(--model cross_modal --video-dir data/wlasl/).

For each .mp4 in data/wlasl/videos/ the script:
  1. Opens the video with OpenCV.
  2. Runs HandLandmarkExtractor on every frame.
  3. Collects all per-frame (63,) vectors where a hand was detected.
  4. Computes mean and std across those frames.
  5. Stores one (mean_landmarks,) entry per video.
  6. Also aggregates a per-word mean by averaging across all videos for that word.

Final output:
  data/wlasl/landmarks.npz
    A compressed NumPy archive where each key is a word gloss (e.g. "hello")
    and each value is the (63,) mean landmark prototype for that word.
    This format is consumed directly by _load_cross_modal_dataset() in
    scripts/train_gpu_ddp.py.

  data/wlasl/landmarks_per_video.npz (diagnostic artifact)
    Same archive but indexed by "{word}/{video_id}" keys, with value (63,).
    Useful for debugging coverage per signer.

Usage:

  # Basic (CPU):
  python scripts/extract_wlasl_landmarks.py

  # Custom video directory and output:
  python scripts/extract_wlasl_landmarks.py \\
      --video-dir data/wlasl/videos/ \\
      --output-dir data/wlasl/

  # GPU-accelerated MediaPipe (requires mediapipe with GPU support):
  python scripts/extract_wlasl_landmarks.py --gpu

Dependencies:
  pip install mediapipe>=0.10 opencv-python>=4.9
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src.* imports work when invoked from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-video landmark extraction
# ---------------------------------------------------------------------------


def _extract_video(
    video_path: Path,
    use_gpu: bool,
) -> tuple[np.ndarray | None, int, int]:
    """Extract the mean hand landmark vector from a single video file.

    Processes every frame sequentially.  Frames where no hand is detected
    are skipped.

    Parameters
    ----------
    video_path:
        Path to a .mp4 video file.
    use_gpu:
        Whether to request GPU-accelerated inference from MediaPipe.
        Has no effect if the mediapipe build does not support GPU.

    Returns
    -------
    mean_landmarks:
        Shape (63,) float32 mean landmark vector across all frames with a
        hand detected, or None if no frames had a hand.
    total_frames:
        Total number of frames in the video.
    hand_frames:
        Number of frames where a hand was detected.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python not installed. Run: pip install 'opencv-python>=4.9'")
        return None, 0, 0

    try:
        from src.data.vision_teacher import HandLandmarkExtractor
    except ImportError as exc:
        logger.error("Could not import HandLandmarkExtractor: %s", exc)
        return None, 0, 0

    # HandLandmarkExtractor does not have a gpu flag; static_image_mode=True can
    # improve quality for non-real-time processing at the cost of speed.
    extractor = HandLandmarkExtractor(
        max_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None, 0, 0

    total_frames = 0
    frame_vectors: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        lm = extractor.extract(frame)
        if lm is not None:
            frame_vectors.append(lm.astype(np.float32))

    cap.release()

    hand_frames = len(frame_vectors)
    if hand_frames == 0:
        return None, total_frames, 0

    arr = np.stack(frame_vectors, axis=0)    # (T, 63)
    mean_lm = arr.mean(axis=0)               # (63,)
    return mean_lm, total_frames, hand_frames


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def extract_all(
    video_dir: str,
    output_dir: str,
    use_gpu: bool,
) -> None:
    """Process all .mp4 files under video_dir and save landmarks.npz.

    Parameters
    ----------
    video_dir:
        Directory containing per-word subdirectories of .mp4 files
        (layout: <video_dir>/<word>/<video_id>.mp4).
    output_dir:
        Directory where landmarks.npz and landmarks_per_video.npz are saved.
    use_gpu:
        Passed through to _extract_video.
    """
    vid_root = Path(video_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(vid_root.rglob("*.mp4"))
    if not video_files:
        logger.error(
            "No .mp4 files found in %s. "
            "Run scripts/download_wlasl.py first.",
            vid_root,
        )
        sys.exit(1)

    logger.info("Found %d video files to process.", len(video_files))

    # Per-video results: {word: {video_id: mean_lm (63,)}}
    per_video: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    total_all_frames = 0
    total_hand_frames = 0
    n_no_hand = 0
    t_start = time.time()

    for i, vf in enumerate(video_files, 1):
        # Derive word and video_id from directory structure.
        # Expected path: <video_dir>/<word>/<video_id>.mp4
        word = vf.parent.name
        video_id = vf.stem

        mean_lm, n_frames, n_hand = _extract_video(vf, use_gpu=use_gpu)
        total_all_frames += n_frames
        total_hand_frames += n_hand

        if mean_lm is not None:
            per_video[word][video_id] = mean_lm
        else:
            n_no_hand += 1

        if i % 100 == 0 or i == len(video_files):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(video_files) - i) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d videos  hand_detected=%d  no_hand=%d  "
                "%.1f vid/s  ETA ~%.0f min",
                i, len(video_files),
                i - n_no_hand, n_no_hand,
                rate, remaining / 60,
            )

    # Coverage per word.
    words_covered = sorted(per_video.keys())
    logger.info("\n--- Coverage per word ---")
    for word in words_covered:
        n_vids = len(per_video[word])
        logger.info("  %-30s  %d video(s) with hands", word, n_vids)

    overall_coverage = (
        100 * (len(video_files) - n_no_hand) / len(video_files)
        if video_files else 0.0
    )
    frame_coverage = (
        100 * total_hand_frames / total_all_frames
        if total_all_frames else 0.0
    )
    logger.info(
        "\nTotal videos processed : %d", len(video_files),
    )
    logger.info("  Videos with hands    : %d (%.1f%%)", len(video_files) - n_no_hand, overall_coverage)
    logger.info("  Videos no hand       : %d", n_no_hand)
    logger.info("  Total frames         : %d", total_all_frames)
    logger.info("  Frames with hands    : %d (%.1f%%)", total_hand_frames, frame_coverage)

    # Build per-word mean prototype (average over all per-video means).
    word_prototypes: dict[str, np.ndarray] = {}
    for word, vids in per_video.items():
        if not vids:
            continue
        stacked = np.stack(list(vids.values()), axis=0)   # (V, 63)
        proto = stacked.mean(axis=0)                        # (63,)
        # L2-normalize the prototype for cleaner cosine similarity at inference.
        norm = np.linalg.norm(proto) + 1e-9
        word_prototypes[word] = (proto / norm).astype(np.float32)

    # Save landmarks.npz (word -> prototype).
    landmarks_path = out_root / "landmarks.npz"
    np.savez_compressed(str(landmarks_path), **word_prototypes)
    logger.info("\nSaved word prototypes to %s (%d words)", landmarks_path, len(word_prototypes))

    # Save per-video diagnostic archive.
    per_video_flat: dict[str, np.ndarray] = {}
    for word, vids in per_video.items():
        for vid_id, lm in vids.items():
            per_video_flat[f"{word}/{vid_id}"] = lm

    per_video_path = out_root / "landmarks_per_video.npz"
    np.savez_compressed(str(per_video_path), **per_video_flat)
    logger.info("Saved per-video landmarks to %s", per_video_path)

    logger.info(
        "\nExtraction complete. Use data/wlasl/landmarks.npz with:\n"
        "  torchrun ... scripts/train_gpu_ddp.py --model cross_modal "
        "--video-dir data/wlasl/"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from downloaded WLASL videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video-dir",
        default="data/wlasl/videos/",
        metavar="PATH",
        help="Directory containing per-word subdirectories of .mp4 files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/wlasl/",
        metavar="PATH",
        help="Directory where landmarks.npz will be written.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help=(
            "Request GPU-accelerated MediaPipe inference (requires a mediapipe "
            "build with GPU support; falls back to CPU silently if unavailable)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args and run landmark extraction."""
    args = parse_args()
    extract_all(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()

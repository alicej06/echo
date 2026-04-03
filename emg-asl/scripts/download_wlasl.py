#!/usr/bin/env python3
"""
Download the WLASL dataset for cross-modal EMG-ASL training.

WLASL (Word-Level American Sign Language) is the largest public ASL video dataset:
  - 21,083 video clips
  - 2,000 ASL words (a massive vocabulary)
  - 119 signers
  - Videos are YouTube clips with precise start/end timestamps
  - License: Researcher Use Only (see https://github.com/dxli94/WLASL)

How cross-modal training works:
  1. Extract hand landmarks from each WLASL video using MediaPipe.
  2. Compute vision embeddings using VisionEncoder (from cross_modal_embedding.py).
  3. Build a class gallery: mean vision embedding per word.
  4. Classify EMG windows by nearest-neighbor lookup in the gallery.
  5. Zero-shot: recognize any of the 2,000 words without EMG recordings for them.

This gives the EMG-ASL model a 2,000-word vocabulary with no additional EMG
data collection needed.

Usage examples:

  # Download all 2,000 words (large -- ~100 GB):
  python scripts/download_wlasl.py --output-dir data/wlasl/

  # Download only the 26 ASL letters and 10 common words (small -- ~500 MB):
  python scripts/download_wlasl.py \\
      --words-only A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,HELLO,THANK_YOU,PLEASE,YES,NO,HELP,WATER,MORE,STOP,GO \\
      --output-dir data/wlasl/

  # Download + extract landmarks immediately:
  python scripts/download_wlasl.py \\
      --words-only A,B,C \\
      --extract-landmarks \\
      --output-dir data/wlasl/

  # Resume an interrupted download:
  python scripts/download_wlasl.py --output-dir data/wlasl/
  (progress is saved to data/wlasl/download_progress.json and skips completed videos)

Dependencies (install once):
  pip install yt-dlp
  pip install mediapipe opencv-python   # only needed with --extract-landmarks
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure src.* imports work when invoked from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)

# Rough average compressed video size in MB (used for disk-space estimates only).
_APPROX_MB_PER_VIDEO = 5.0

# The 36 classes built into the EMG-ASL system (26 letters + 10 words).
_DEFAULT_SUBSET: list[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "hello", "thank you", "please", "yes", "no", "help", "water", "more",
    "stop", "go",
]


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------


def _check_ytdlp() -> None:
    """Raise a clear error if yt-dlp is not installed."""
    if shutil.which("yt-dlp") is None:
        sys.exit(
            "ERROR: yt-dlp is not installed.\n"
            "Install it with:  pip install yt-dlp\n"
            "Then re-run this script."
        )


def _download_video(
    url: str,
    output_path: Path,
    start_sec: Optional[float],
    end_sec: Optional[float],
    timeout_sec: int = 120,
) -> bool:
    """Download a single video clip using yt-dlp.

    The clip is trimmed to [start_sec, end_sec] when both are provided.

    Parameters
    ----------
    url:
        YouTube URL for the video.
    output_path:
        Full destination path including filename (e.g. .../videos/hello/12345.mp4).
    start_sec:
        Start offset within the YouTube video in seconds, or None to keep full video.
    end_sec:
        End offset within the YouTube video in seconds, or None to keep full video.
    timeout_sec:
        Maximum seconds to wait before killing the download process.

    Returns
    -------
    bool
        True on success, False on failure (non-zero exit or timeout).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", str(output_path),
    ]

    if start_sec is not None and end_sec is not None:
        # --download-sections uses the format "*<start>-<end>".
        cmd += ["--download-sections", f"*{start_sec}-{end_sec}"]

    cmd.append(url)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("Timeout downloading %s", url)
        return False
    except Exception as exc:
        logger.warning("Exception downloading %s: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# Progress checkpointing
# ---------------------------------------------------------------------------


def _load_progress(progress_path: Path) -> set[str]:
    """Load the set of already-downloaded video IDs from the checkpoint file."""
    if not progress_path.exists():
        return set()
    try:
        with open(progress_path) as fh:
            data = json.load(fh)
        return set(data.get("completed", []))
    except Exception:
        return set()


def _save_progress(progress_path: Path, completed: set[str]) -> None:
    """Persist the set of completed video IDs to disk."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w") as fh:
        json.dump({"completed": sorted(completed)}, fh, indent=2)


# ---------------------------------------------------------------------------
# WLASL metadata parsing
# ---------------------------------------------------------------------------


def _fetch_wlasl_metadata() -> list[dict]:
    """Download and parse the WLASL v0.3 JSON metadata.

    Returns
    -------
    list[dict]
        The raw list of word entries from the JSON.  Each entry has the form:
        {"gloss": "hello", "instances": [{"video_id": ..., "url": ...,
        "start_time": ..., "end_time": ...}, ...]}
    """
    import urllib.request

    logger.info("Fetching WLASL metadata from %s", WLASL_JSON_URL)
    with urllib.request.urlopen(WLASL_JSON_URL, timeout=30) as resp:
        data = json.loads(resp.read())
    logger.info("Metadata loaded: %d word entries", len(data))
    return data


def _build_download_list(
    metadata: list[dict],
    words_filter: Optional[set[str]],
    max_per_word: int,
) -> list[dict]:
    """Convert the WLASL metadata into a flat list of download tasks.

    Parameters
    ----------
    metadata:
        Raw word-level entries from WLASL JSON.
    words_filter:
        If not None, only include words whose gloss is in this set
        (case-insensitive comparison).
    max_per_word:
        Maximum number of videos to include per word.

    Returns
    -------
    list[dict]
        Each item: {"word": str, "video_id": str, "url": str,
                    "start_time": float or None, "end_time": float or None}
    """
    tasks: list[dict] = []

    for entry in metadata:
        gloss: str = entry.get("gloss", "").strip()
        if words_filter is not None:
            if gloss.lower() not in {w.lower() for w in words_filter}:
                continue

        instances = entry.get("instances", [])
        for inst in instances[:max_per_word]:
            video_id = str(inst.get("video_id", ""))
            url = inst.get("url", "")
            if not url:
                continue
            tasks.append({
                "word": gloss,
                "video_id": video_id,
                "url": url,
                "start_time": inst.get("start_time", None),
                "end_time": inst.get("end_time", None),
            })

    return tasks


# ---------------------------------------------------------------------------
# Landmark extraction (called with --extract-landmarks)
# ---------------------------------------------------------------------------


def _extract_landmarks_for_video(
    video_path: Path,
    landmark_path: Path,
) -> bool:
    """Extract per-frame hand landmarks from a single video file.

    Uses HandLandmarkExtractor from src/data/vision_teacher.py to process
    every frame.  The mean and standard deviation of landmarks across all
    frames where a hand is detected are saved as a (2, 63) array (row 0 = mean,
    row 1 = std).

    Parameters
    ----------
    video_path:
        Path to the .mp4 video file.
    landmark_path:
        Output .npy file path.

    Returns
    -------
    bool
        True if at least one frame with a hand was detected, False otherwise.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python is required for landmark extraction. Run: pip install opencv-python>=4.9")
        return False

    try:
        from src.data.vision_teacher import HandLandmarkExtractor
    except ImportError as exc:
        logger.error("Could not import HandLandmarkExtractor: %s", exc)
        return False

    extractor = HandLandmarkExtractor(max_hands=1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return False

    frame_landmarks: list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lm = extractor.extract(frame)
        if lm is not None:
            frame_landmarks.append(lm)

    cap.release()

    if not frame_landmarks:
        return False

    import numpy as np
    arr = np.stack(frame_landmarks, axis=0)          # (T, 63)
    mean_lm = arr.mean(axis=0)                        # (63,)
    std_lm = arr.std(axis=0)                          # (63,)
    result = np.stack([mean_lm, std_lm], axis=0)      # (2, 63)

    landmark_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(str(landmark_path), result)
    return True


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------


def download_wlasl(
    output_dir: str,
    words_only: Optional[str],
    max_videos_per_word: int,
    extract_landmarks: bool,
    jobs: int,
) -> None:
    """Download WLASL videos and optionally extract hand landmarks.

    Parameters
    ----------
    output_dir:
        Root directory for all downloaded data (e.g. ``data/wlasl/``).
    words_only:
        Comma-separated list of word glosses to download.  Downloads all
        2,000 words when None.
    max_videos_per_word:
        Cap on videos downloaded per word.
    extract_landmarks:
        Whether to run HandLandmarkExtractor on each downloaded video.
    jobs:
        Number of parallel download workers.
    """
    _check_ytdlp()

    out = Path(output_dir)
    videos_dir = out / "videos"
    landmarks_dir = out / "landmarks"
    progress_path = out / "download_progress.json"

    out.mkdir(parents=True, exist_ok=True)

    words_filter: Optional[set[str]] = None
    if words_only:
        words_filter = {w.strip() for w in words_only.split(",") if w.strip()}
        logger.info("Filtering to %d words: %s", len(words_filter), sorted(words_filter))

    # Fetch metadata.
    metadata = _fetch_wlasl_metadata()

    # Build flat task list.
    tasks = _build_download_list(metadata, words_filter, max_videos_per_word)

    n_words = len({t["word"] for t in tasks})
    est_gb = len(tasks) * _APPROX_MB_PER_VIDEO / 1024
    logger.info(
        "Download plan: %d videos across %d words (estimated ~%.1f GB)",
        len(tasks), n_words, est_gb,
    )

    # Load checkpoint.
    completed: set[str] = _load_progress(progress_path)
    skipped = sum(1 for t in tasks if t["video_id"] in completed)
    if skipped:
        logger.info("Resuming: %d videos already downloaded, skipping.", skipped)

    pending = [t for t in tasks if t["video_id"] not in completed]
    logger.info("%d videos to download.", len(pending))

    if not pending:
        logger.info("Nothing to download -- all videos already present.")
    else:
        # Parallel download with progress tracking.
        n_success = 0
        n_fail = 0
        t_start = time.time()

        def _worker(task: dict) -> tuple[str, bool]:
            vid_id: str = task["video_id"]
            word: str = task["word"]
            url: str = task["url"]
            dest = videos_dir / word / f"{vid_id}.mp4"

            if dest.exists():
                return vid_id, True

            ok = _download_video(
                url=url,
                output_path=dest,
                start_sec=task["start_time"],
                end_sec=task["end_time"],
            )
            return vid_id, ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_worker, t): t for t in pending}
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                vid_id, ok = fut.result()
                if ok:
                    n_success += 1
                    completed.add(vid_id)
                else:
                    n_fail += 1

                if i % 50 == 0 or i == len(pending):
                    elapsed = time.time() - t_start
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (len(pending) - i) / rate if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d  success=%d  fail=%d  "
                        "%.1f vid/s  ETA ~%.0f min",
                        i, len(pending), n_success, n_fail,
                        rate, remaining / 60,
                    )
                    _save_progress(progress_path, completed)

        _save_progress(progress_path, completed)
        logger.info(
            "Download complete: %d success, %d failed. Progress saved to %s",
            n_success, n_fail, progress_path,
        )

    # Landmark extraction.
    if extract_landmarks:
        logger.info("Extracting hand landmarks from downloaded videos ...")
        video_files = list(videos_dir.rglob("*.mp4"))
        n_ok = 0
        n_no_hand = 0
        for vf in video_files:
            # Preserve word subdirectory structure.
            rel = vf.relative_to(videos_dir)
            lm_path = landmarks_dir / rel.with_suffix(".npy")
            if lm_path.exists():
                n_ok += 1
                continue
            success = _extract_landmarks_for_video(vf, lm_path)
            if success:
                n_ok += 1
            else:
                n_no_hand += 1

        total = len(video_files)
        pct = 100 * n_ok / total if total else 0
        logger.info(
            "Landmark extraction done: %d/%d videos had detectable hands (%.1f%%), "
            "%d had no hand detected.",
            n_ok, total, pct, n_no_hand,
        )
        logger.info("Landmarks saved to %s", landmarks_dir)
    else:
        logger.info(
            "Skipping landmark extraction. "
            "Run scripts/extract_wlasl_landmarks.py or re-run with --extract-landmarks."
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the WLASL ASL video dataset for cross-modal EMG-ASL training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        default="data/wlasl/",
        metavar="PATH",
        help="Root directory where videos and landmarks will be saved.",
    )
    parser.add_argument(
        "--words-only",
        default=None,
        metavar="WORD1,WORD2,...",
        help=(
            "Comma-separated list of WLASL glosses to download.  "
            "Downloads all 2,000 words when omitted (large)."
        ),
    )
    parser.add_argument(
        "--max-videos-per-word",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of videos to download per word.",
    )
    parser.add_argument(
        "--extract-landmarks",
        action="store_true",
        default=False,
        help=(
            "After downloading, extract MediaPipe hand landmarks from each video "
            "and save per-video .npy files.  Requires mediapipe and opencv-python."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel download workers.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args and run the download pipeline."""
    args = parse_args()
    download_wlasl(
        output_dir=args.output_dir,
        words_only=args.words_only,
        max_videos_per_word=args.max_videos_per_word,
        extract_landmarks=args.extract_landmarks,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()

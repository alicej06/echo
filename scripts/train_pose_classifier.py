#!/usr/bin/env python3
"""
Train the ASL hand pose classifier used by VisionTeacher.

The classifier maps 63-dimensional MediaPipe hand landmark vectors
to ASL letter labels (A-Z). It is the 'teacher' component in the
vision-to-EMG cross-modal labeling pipeline.

Usage:
  # Collect training data from webcam (5 seconds per letter):
  python scripts/train_pose_classifier.py --collect

  # Train classifier from saved landmarks:
  python scripts/train_pose_classifier.py --train

  # Collect and train in one step:
  python scripts/train_pose_classifier.py --collect --train

  # Evaluate an existing classifier:
  python scripts/train_pose_classifier.py --evaluate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path so local src imports work whether this script is
# invoked from the project root or from the scripts/ subdirectory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.vision_teacher import (  # noqa: E402
    HandLandmarkExtractor,
    SimpleASLPoseClassifier,
)
from src.utils.constants import ASL_LETTERS  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_LANDMARKS_DIR = _PROJECT_ROOT / "data" / "pose_landmarks"
_MODELS_DIR = _PROJECT_ROOT / "models"
_CLASSIFIER_PATH = _MODELS_DIR / "pose_classifier.joblib"

# ---------------------------------------------------------------------------
# Collection parameters
# ---------------------------------------------------------------------------

_COLLECT_SECONDS: int = 5          # recording window per letter
_MIN_DETECTION_CONF: float = 0.5   # hand detection confidence threshold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_cv2() -> None:
    """Abort with a clear error message if OpenCV is not installed."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[ERROR] opencv-python is required for webcam collection.")
        print("  pip install 'opencv-python>=4.9'")
        sys.exit(1)


def _check_mediapipe() -> None:
    """Abort with a clear error message if mediapipe is not installed."""
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print("[ERROR] mediapipe is required for hand landmark extraction.")
        print("  pip install 'mediapipe>=0.10'")
        sys.exit(1)


def _check_sklearn() -> None:
    """Abort with a clear error message if scikit-learn is not installed."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("[ERROR] scikit-learn is required to fit the pose classifier.")
        print("  pip install 'scikit-learn>=1.4'")
        sys.exit(1)


def _check_joblib() -> None:
    """Abort if joblib is not installed."""
    try:
        import joblib  # noqa: F401
    except ImportError:
        print("[ERROR] joblib is required to save the classifier.")
        print("  pip install joblib")
        sys.exit(1)


# ---------------------------------------------------------------------------
# collect_landmarks
# ---------------------------------------------------------------------------


def collect_landmarks(force: bool = False) -> None:
    """
    Open the webcam and walk through each letter A-Z, collecting MediaPipe
    hand landmark vectors for ``_COLLECT_SECONDS`` seconds per letter.

    Collected data is saved to ``data/pose_landmarks/{letter}.npy`` as an
    array of shape (N, 63).

    Parameters
    ----------
    force:
        When True, re-collect all letters even if .npy files already exist.
        When False (default), letters that already have saved data are skipped.
    """
    _check_cv2()
    _check_mediapipe()

    import cv2  # noqa: WPS433

    _LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    extractor = HandLandmarkExtractor(
        max_hands=1,
        min_detection_confidence=_MIN_DETECTION_CONF,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        extractor.close()
        print("[ERROR] Could not open webcam (device index 0).")
        print("  Make sure a webcam is connected and not in use by another app.")
        sys.exit(1)

    print("=" * 60)
    print("ASL Pose Classifier -- Data Collection")
    print("=" * 60)
    print(f"Recording {_COLLECT_SECONDS} seconds per letter (A-Z).")
    print("Show your hand clearly in the frame, palm facing the camera.")
    print("Press  q  at any time to abort.\n")

    try:
        for letter in ASL_LETTERS:
            out_path = _LANDMARKS_DIR / f"{letter}.npy"

            if out_path.exists() and not force:
                existing = np.load(str(out_path))
                print(
                    f"  [{letter}] Skipping -- {existing.shape[0]} samples already saved"
                    f" ({out_path.name}). Use --force to re-collect."
                )
                continue

            # ---- countdown + recording loop --------------------------------
            collected: list[np.ndarray] = []
            record_start: float | None = None
            phase = "countdown"  # "countdown" | "recording" | "done"
            phase_start = time.monotonic()

            print(f"\n  [{letter}]  Get ready to sign  '{letter}'  ...")

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[WARNING] Failed to read frame from webcam.")
                    break

                # Mirror the frame so the user sees a natural reflection.
                display = cv2.flip(frame, 1)
                now = time.monotonic()

                if phase == "countdown":
                    elapsed = now - phase_start
                    countdown_val = max(0, 3 - int(elapsed))
                    msg_top = f"Sign: {letter}"
                    msg_mid = f"Starting in {countdown_val}..." if countdown_val > 0 else "GO!"
                    if elapsed >= 3.0:
                        phase = "recording"
                        record_start = now

                elif phase == "recording":
                    elapsed = now - record_start  # type: ignore[operator]
                    remaining = max(0.0, _COLLECT_SECONDS - elapsed)
                    msg_top = f"Sign: {letter}  --  HOLD STILL"
                    msg_mid = f"Recording... {remaining:.1f}s remaining"

                    # Attempt landmark extraction on the unflipped frame so
                    # coordinates are consistent with the model's expectations.
                    vec = extractor.extract(frame)
                    if vec is not None:
                        collected.append(vec)

                    if elapsed >= _COLLECT_SECONDS:
                        phase = "done"

                else:  # phase == "done"
                    msg_top = f"Sign: {letter}  --  Done"
                    msg_mid = f"Collected {len(collected)} frames"

                # ---- overlay text ------------------------------------------
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(display, msg_top, (10, 30), font, 0.9, (255, 255, 255), 2)
                cv2.putText(display, msg_mid, (10, 65), font, 0.7, (0, 230, 0), 2)

                if phase == "recording":
                    bar_w = int(
                        display.shape[1]
                        * min(1.0, (now - record_start) / _COLLECT_SECONDS)  # type: ignore[operator]
                    )
                    cv2.rectangle(
                        display,
                        (0, display.shape[0] - 10),
                        (bar_w, display.shape[0]),
                        (0, 200, 0),
                        -1,
                    )

                cv2.imshow("ASL Pose Collector", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[ABORTED] Collection cancelled by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    extractor.close()
                    return

                if phase == "done":
                    # Brief pause so the user can see the "Done" message.
                    cv2.waitKey(400)
                    break

            # ---- save results ----------------------------------------------
            if collected:
                arr = np.stack(collected, axis=0)  # shape (N, 63)
                np.save(str(out_path), arr)
                print(f"  [{letter}]  Saved {arr.shape[0]} landmark vectors -> {out_path}")
            else:
                print(
                    f"  [{letter}]  WARNING: no hand detected during recording window."
                    f" Data NOT saved. Re-run with --force to retry."
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    print("\nCollection complete.")
    print(f"Landmark files saved to: {_LANDMARKS_DIR}")
    print("Run  python scripts/train_pose_classifier.py --train  to fit the classifier.\n")


# ---------------------------------------------------------------------------
# train_classifier
# ---------------------------------------------------------------------------


def train_classifier() -> None:
    """
    Load all saved .npy landmark files from ``data/pose_landmarks/``, fit a
    SimpleASLPoseClassifier, print per-class accuracy, and save the trained
    model to ``models/pose_classifier.joblib``.
    """
    _check_sklearn()
    _check_joblib()

    from sklearn.model_selection import train_test_split  # noqa: WPS433
    from sklearn.metrics import accuracy_score  # noqa: WPS433

    print("=" * 60)
    print("ASL Pose Classifier -- Training")
    print("=" * 60)

    if not _LANDMARKS_DIR.exists():
        print(f"[ERROR] Landmark directory not found: {_LANDMARKS_DIR}")
        print("  Run  python scripts/train_pose_classifier.py --collect  first.")
        sys.exit(1)

    # ---- load data ---------------------------------------------------------
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    letters_found: list[str] = []
    letters_missing: list[str] = []

    for letter in ASL_LETTERS:
        npy_path = _LANDMARKS_DIR / f"{letter}.npy"
        if npy_path.exists():
            arr = np.load(str(npy_path))
            if arr.ndim == 2 and arr.shape[1] == 63 and len(arr) > 0:
                X_parts.append(arr)
                y_parts.append(np.full(len(arr), letter, dtype=object))
                letters_found.append(letter)
            else:
                print(
                    f"  [WARNING] {npy_path.name} has unexpected shape {arr.shape},"
                    " skipping."
                )
                letters_missing.append(letter)
        else:
            letters_missing.append(letter)

    if not letters_found:
        print("[ERROR] No landmark data found. Run --collect first.")
        sys.exit(1)

    print(f"  Letters with data ({len(letters_found)}): {' '.join(letters_found)}")
    if letters_missing:
        print(
            f"  Letters without data ({len(letters_missing)}): {' '.join(letters_missing)}"
        )
        print("  The classifier will only cover the letters listed above.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    print(f"\n  Total samples: {len(X)}  |  Features: {X.shape[1]}  |  Classes: {len(letters_found)}")

    # ---- train / val split -------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples  |  Val: {len(X_val)} samples")

    # ---- fit ---------------------------------------------------------------
    print("\n  Fitting SimpleASLPoseClassifier (SVC RBF, C=10) ...")
    t0 = time.monotonic()
    clf = SimpleASLPoseClassifier()
    clf.fit(X_train, y_train)
    elapsed = time.monotonic() - t0
    print(f"  Fit complete in {elapsed:.1f}s")

    # ---- per-class accuracy ------------------------------------------------
    y_pred = np.array([clf.predict_proba(x)[0] for x in X_val])
    overall_acc = accuracy_score(y_val, y_pred)

    print("\n  Per-class accuracy on validation set:")
    print(f"  {'Letter':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-" * 38)
    for letter in letters_found:
        mask = y_val == letter
        if mask.sum() == 0:
            continue
        correct = (y_pred[mask] == letter).sum()
        total = mask.sum()
        acc = correct / total
        bar = "#" * int(acc * 20)
        print(f"  {letter:<8} {correct:>8} {total:>8} {acc:>9.1%}  {bar}")

    print("  " + "-" * 38)
    print(f"  {'OVERALL':<8} {'':>8} {len(X_val):>8} {overall_acc:>9.1%}")

    # ---- save model --------------------------------------------------------
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf.save(str(_CLASSIFIER_PATH))
    print(f"\n  Saved -> {_CLASSIFIER_PATH}")
    print(
        "\n  To use this classifier:"
        "\n    python scripts/auto_label_session.py --webcam"
        f" --classifier {_CLASSIFIER_PATH}\n"
    )


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------


def evaluate_classifier() -> None:
    """
    Load an existing classifier from ``models/pose_classifier.joblib``, load
    all saved landmark data, and print a compact ASCII confusion matrix plus
    per-class F1 scores.
    """
    _check_sklearn()
    _check_joblib()

    from sklearn.metrics import (  # noqa: WPS433
        classification_report,
        confusion_matrix,
    )

    print("=" * 60)
    print("ASL Pose Classifier -- Evaluation")
    print("=" * 60)

    if not _CLASSIFIER_PATH.exists():
        print(f"[ERROR] Classifier not found: {_CLASSIFIER_PATH}")
        print("  Run  python scripts/train_pose_classifier.py --train  first.")
        sys.exit(1)

    clf = SimpleASLPoseClassifier()
    clf.load(str(_CLASSIFIER_PATH))
    print(f"  Loaded classifier from {_CLASSIFIER_PATH}")

    if not _LANDMARKS_DIR.exists():
        print(f"[ERROR] Landmark directory not found: {_LANDMARKS_DIR}")
        sys.exit(1)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    letters_found: list[str] = []

    for letter in ASL_LETTERS:
        npy_path = _LANDMARKS_DIR / f"{letter}.npy"
        if npy_path.exists():
            arr = np.load(str(npy_path))
            if arr.ndim == 2 and arr.shape[1] == 63 and len(arr) > 0:
                X_parts.append(arr)
                y_parts.append(np.full(len(arr), letter, dtype=object))
                letters_found.append(letter)

    if not X_parts:
        print("[ERROR] No landmark data found to evaluate against.")
        sys.exit(1)

    X = np.concatenate(X_parts, axis=0)
    y_true = np.concatenate(y_parts, axis=0)

    print(f"  Evaluating on {len(X)} samples across {len(letters_found)} letters ...\n")

    y_pred = np.array([clf.predict_proba(x)[0] for x in X])

    # ---- per-class F1 report -----------------------------------------------
    print(classification_report(y_true, y_pred, labels=letters_found, zero_division=0))

    # ---- compact ASCII confusion matrix ------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=letters_found)
    n = len(letters_found)
    cell_w = 4

    print("  Confusion matrix (rows = true, cols = predicted):")
    header = "     " + "".join(f"{l:>{cell_w}}" for l in letters_found)
    print("  " + header)
    print("  " + "-" * len(header))
    for i, letter in enumerate(letters_found):
        row_vals = "".join(
            f"{cm[i, j]:>{cell_w}}" if cm[i, j] > 0 else f"{'·':>{cell_w}}"
            for j in range(n)
        )
        print(f"  {letter:<4} {row_vals}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Open the webcam and record 5 seconds of landmarks per letter.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Load saved landmarks and fit the pose classifier.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Load existing classifier and print confusion matrix + per-class F1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --collect: re-collect letters that already have saved data, "
            "overwriting existing .npy files."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not (args.collect or args.train or args.evaluate):
        parser.print_help()
        print(
            "\n[ERROR] Specify at least one mode: --collect, --train, or --evaluate."
        )
        sys.exit(1)

    if args.collect:
        collect_landmarks(force=args.force)

    if args.train:
        train_classifier()

    if args.evaluate:
        evaluate_classifier()


if __name__ == "__main__":
    main()

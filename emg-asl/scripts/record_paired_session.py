#!/usr/bin/env python3
"""
Record simultaneous webcam video + EMG for cross-modal training.

This script records both modalities in lockstep using a shared wall-clock
timestamp. The two streams can later be aligned using sync_labels_to_emg()
from src/data/vision_teacher.py for automatic label generation.

Usage:
  # Hardware mode (BLE armband required):
  python scripts/record_paired_session.py \\
      --participant P001 \\
      --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z \\
      --reps 5

  # Synthetic mode (no hardware, useful for testing):
  python scripts/record_paired_session.py \\
      --participant TEST001 \\
      --synthetic \\
      --labels A B C \\
      --reps 3

  # Live webcam test (no EMG, just checks camera + MediaPipe):
  python scripts/record_paired_session.py --test-webcam

Outputs:
  data/raw/P001_20260301_143022.csv         (EMG session CSV, same format as record_session.py)
  data/raw/P001_20260301_143022_video.mp4   (corresponding webcam recording)
  data/raw/P001_20260301_143022_labels.csv  (vision-derived labels with timestamps)

All three files share the same timestamp prefix for easy pairing.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import struct
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.*`` imports work when
# invoked as ``python scripts/record_paired_session.py`` from the project
# root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.constants import (
    ASL_LABELS,
    ASL_LETTERS,
    BLE_DEVICE_NAME,
    BLE_EMG_CHAR_UUID,
    N_CHANNELS,
    SAMPLE_RATE,
)
from src.data.vision_teacher import (
    HandLandmarkExtractor,
    sync_labels_to_emg,
)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

# myo-python imports -- only needed in hardware mode
# Hardware mode uses myo-python + MyoConnect (same as record_session.py).
# bleak is no longer used; the MYO Armband is accessed via myo-python on laptop,
# or via direct BLE from the React Native mobile app.
try:
    import myo as _myo_sdk
    _MYO_AVAILABLE = True
except ImportError:
    _MYO_AVAILABLE = False

# Legacy bleak import (kept for graceful ImportError handling in older envs)
_BLEAK_AVAILABLE = False

# ---------------------------------------------------------------------------
# BLE packet decoding (mirrored from record_session.py)
# ---------------------------------------------------------------------------

_BYTES_PER_PACKET: int = 16
_PACKET_FMT: str = ">8h"


def _decode_packet(data: bytearray) -> List[float]:
    if len(data) < _BYTES_PER_PACKET:
        data = data + bytearray(_BYTES_PER_PACKET - len(data))
    raw = struct.unpack_from(_PACKET_FMT, bytes(data[:_BYTES_PER_PACKET]))
    return [r / 1000.0 for r in raw]


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

# Each EMG sample: (wall_clock_ms, [ch1..ch8], label)
EMGSample = Tuple[float, List[float], str]
# Each video frame index: (wall_clock_ms, frame_idx)
VideoTick = Tuple[float, int]
# Each vision label: (wall_clock_ms, label, confidence)
VisionLabel = Tuple[float, str, float]


# ---------------------------------------------------------------------------
# Video recording thread
# ---------------------------------------------------------------------------

class VideoThread(threading.Thread):
    """
    Captures frames from the webcam, writes them to an MP4 file, and appends
    (wall_clock_ms, frame_idx) ticks to a shared list for later alignment.

    wall_clock_ms is computed as (time.monotonic() * 1000.0) relative to the
    session epoch handed in at construction time. This matches the EMG thread's
    time base exactly.
    """

    def __init__(
        self,
        camera_index: int,
        output_path: Path,
        session_epoch_ms: float,
        frame_ticks: List[VideoTick],
        vision_labels: List[VisionLabel],
        start_event: threading.Event,
        stop_event: threading.Event,
        recording_event: threading.Event,
        current_label_ref: List[str],
    ) -> None:
        super().__init__(name="VideoThread", daemon=True)
        self._camera_index = camera_index
        self._output_path = output_path
        self._epoch_ms = session_epoch_ms
        self._frame_ticks = frame_ticks
        self._vision_labels = vision_labels
        self._start_event = start_event
        self._stop_event = stop_event
        self._recording_event = recording_event
        self._current_label_ref = current_label_ref

        self.fps: float = 30.0
        self.frame_width: int = 640
        self.frame_height: int = 480
        self.error: Optional[str] = None

        self._cap: Optional["cv2.VideoCapture"] = None  # noqa: F821
        self._writer: Optional["cv2.VideoWriter"] = None  # noqa: F821
        self._extractor: Optional[HandLandmarkExtractor] = None

    def run(self) -> None:
        if not _CV2_AVAILABLE:
            self.error = "opencv-python is not installed."
            self._start_event.set()
            return

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self.error = f"Could not open camera index {self._camera_index}."
            self._start_event.set()
            return

        # Query actual camera properties
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if actual_fps and actual_fps > 1.0:
            self.fps = actual_fps
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        if _MP_AVAILABLE:
            try:
                self._extractor = HandLandmarkExtractor()
            except Exception:
                self._extractor = None

        frame_idx = 0
        self._start_event.set()

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                break

            ts_ms = time.monotonic() * 1000.0 - self._epoch_ms
            self._frame_ticks.append((ts_ms, frame_idx))

            # Extract landmarks and store vision label if classifier is available
            if self._extractor is not None and self._recording_event.is_set():
                landmarks = self._extractor.extract(frame)
                if landmarks is not None and _NP_AVAILABLE:
                    # Without a trained classifier we use UNKNOWN at 0.0 confidence
                    # so the ticks are still recorded for timing diagnostics.
                    label = self._current_label_ref[0]
                    self._vision_labels.append((ts_ms, label, 0.0))

            self._writer.write(frame)
            frame_idx += 1

        self._cap.release()
        self._writer.release()
        if self._extractor is not None:
            self._extractor.close()

    def get_preview_frame(self) -> Optional["np.ndarray"]:  # noqa: F821
        """Read one frame from the camera without advancing the file writer."""
        if self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                return frame
        return None


# ---------------------------------------------------------------------------
# EMG recording thread (BLE hardware path)
# ---------------------------------------------------------------------------

class EMGThread(threading.Thread):
    """
    Connects to the BLE armband, receives notifications, and appends
    (wall_clock_ms, [ch1..ch8], label) tuples to a shared list.

    The thread runs an inner asyncio event loop so myo-python's async API works
    inside a standard threading.Thread.
    """

    def __init__(
        self,
        device_name: str,
        session_epoch_ms: float,
        emg_samples: List[EMGSample],
        start_event: threading.Event,
        stop_event: threading.Event,
        recording_event: threading.Event,
        current_label_ref: List[str],
    ) -> None:
        super().__init__(name="EMGThread", daemon=True)
        self._device_name = device_name
        self._epoch_ms = session_epoch_ms
        self._emg_samples = emg_samples
        self._start_event = start_event
        self._stop_event = stop_event
        self._recording_event = recording_event
        self._current_label_ref = current_label_ref
        self.error: Optional[str] = None

    def _on_notification(self, _sender: int, data: bytearray) -> None:
        if not self._recording_event.is_set():
            return
        ts_ms = time.monotonic() * 1000.0 - self._epoch_ms
        channels = _decode_packet(data)
        label = self._current_label_ref[0]
        self._emg_samples.append((ts_ms, channels, label))

    async def _ble_loop(self) -> None:
        if not _MYO_AVAILABLE:
            self.error = "myo-python is not installed. Run: pip install myo-python>=0.2.1 (requires MYO SDK + MyoConnect)"
            self._start_event.set()
            return

        print(f"  [EMG] Scanning for '{self._device_name}' ... (up to 10 s)")
        device = await BleakScanner.find_device_by_name(self._device_name, timeout=10.0)
        if device is None:
            self.error = (
                f"BLE device '{self._device_name}' not found. "
                "Ensure the band is powered on and advertising."
            )
            self._start_event.set()
            return

        print(f"  [EMG] Found device: {device.name}  [{device.address}]")
        async with BleakClient(device) as client:
            if not client.is_connected:
                self.error = f"Failed to connect to {device.address}."
                self._start_event.set()
                return

            print(f"  [EMG] Connected to {device.address}")
            await client.start_notify(BLE_EMG_CHAR_UUID, self._on_notification)
            print(f"  [EMG] Subscribed to EMG characteristic\n")
            self._start_event.set()

            # Block until the main thread signals stop
            while not self._stop_event.is_set():
                await asyncio.sleep(0.01)

            await client.stop_notify(BLE_EMG_CHAR_UUID)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ble_loop())
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Synthetic EMG thread (no hardware required)
# ---------------------------------------------------------------------------

class SyntheticEMGThread(threading.Thread):
    """
    Replays generate_session() data at real-time wall-clock speed (200 Hz).
    Cycles through labels as the main thread advances them.
    """

    def __init__(
        self,
        labels: List[str],
        reps: int,
        hold_duration: float,
        session_epoch_ms: float,
        emg_samples: List[EMGSample],
        start_event: threading.Event,
        stop_event: threading.Event,
        recording_event: threading.Event,
        current_label_ref: List[str],
    ) -> None:
        super().__init__(name="SyntheticEMGThread", daemon=True)
        self._labels = labels
        self._reps = reps
        self._hold_duration = hold_duration
        self._epoch_ms = session_epoch_ms
        self._emg_samples = emg_samples
        self._start_event = start_event
        self._stop_event = stop_event
        self._recording_event = recording_event
        self._current_label_ref = current_label_ref
        self.error: Optional[str] = None

    def run(self) -> None:
        from src.data.synthetic import generate_session

        # Pre-generate data for each label so we are not blocking on numpy
        # during recording.
        print("  [EMG-Synth] Pre-generating synthetic data...")
        synth_data = {}
        for lbl in self._labels:
            df = generate_session(
                label=lbl,
                n_reps=self._reps,
                fs=SAMPLE_RATE,
                hold_duration_s=self._hold_duration,
                rest_duration_s=0.5,
            )
            synth_data[lbl] = df

        print("  [EMG-Synth] Ready.\n")
        self._start_event.set()

        interval_s = 1.0 / SAMPLE_RATE

        while not self._stop_event.is_set():
            if not self._recording_event.is_set():
                time.sleep(0.005)
                continue

            label = self._current_label_ref[0]
            if label not in synth_data:
                time.sleep(0.005)
                continue

            df = synth_data[label]
            ch_cols = [f"ch{i+1}" for i in range(N_CHANNELS)]

            for _, row in df.iterrows():
                if self._stop_event.is_set():
                    break
                if not self._recording_event.is_set():
                    break
                if self._current_label_ref[0] != label:
                    break

                ts_ms = time.monotonic() * 1000.0 - self._epoch_ms
                channels = [float(row[c]) for c in ch_cols]
                self._emg_samples.append((ts_ms, channels, label))
                time.sleep(interval_s)


# ---------------------------------------------------------------------------
# Blank-frame webcam stand-in for --no-webcam mode
# ---------------------------------------------------------------------------

class BlankVideoThread(threading.Thread):
    """
    Writes blank (black) frames to a video file at a fixed rate. Used when
    --synthetic and --no-webcam are both set.
    """

    def __init__(
        self,
        output_path: Path,
        session_epoch_ms: float,
        frame_ticks: List[VideoTick],
        start_event: threading.Event,
        stop_event: threading.Event,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__(name="BlankVideoThread", daemon=True)
        self._output_path = output_path
        self._epoch_ms = session_epoch_ms
        self._frame_ticks = frame_ticks
        self._start_event = start_event
        self._stop_event = stop_event
        self.fps = fps
        self.frame_width = width
        self.frame_height = height
        self.error: Optional[str] = None

    def run(self) -> None:
        if not _CV2_AVAILABLE or not _NP_AVAILABLE:
            self.error = "opencv-python and numpy are required even in --no-webcam mode."
            self._start_event.set()
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )
        blank = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        interval_s = 1.0 / self.fps
        frame_idx = 0
        self._start_event.set()

        while not self._stop_event.is_set():
            ts_ms = time.monotonic() * 1000.0 - self._epoch_ms
            self._frame_ticks.append((ts_ms, frame_idx))
            writer.write(blank)
            frame_idx += 1
            time.sleep(interval_s)

        writer.release()


# ---------------------------------------------------------------------------
# Label-file writer
# ---------------------------------------------------------------------------

def _write_labels_csv(
    path: Path,
    vision_labels: List[VisionLabel],
) -> None:
    """Write per-frame vision labels to a CSV file."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_ms", "label", "vision_confidence"])
        for ts_ms, label, conf in vision_labels:
            writer.writerow([f"{ts_ms:.3f}", label, f"{conf:.4f}"])


# ---------------------------------------------------------------------------
# EMG CSV writer
# ---------------------------------------------------------------------------

def _write_emg_csv(
    path: Path,
    emg_samples: List[EMGSample],
) -> None:
    """Write EMG samples to a CSV file in the same format as record_session.py."""
    channel_headers = [f"ch{i+1}" for i in range(N_CHANNELS)]
    fieldnames = ["timestamp_ms"] + channel_headers + ["label"]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ts_ms, channels, label in emg_samples:
            row: dict = {
                "timestamp_ms": f"{ts_ms:.3f}",
                "label": label,
            }
            for i, val in enumerate(channels):
                row[f"ch{i+1}"] = f"{val:.6f}"
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: "np.ndarray",
    label: str,
    rep: int,
    total_reps: int,
    label_idx: int,
    total_labels: int,
    phase: str,
    countdown: Optional[int],
    recording: bool,
    extractor: Optional[HandLandmarkExtractor],
) -> "np.ndarray":
    """Draw recording overlay and live landmarks onto a BGR frame."""
    if not _CV2_AVAILABLE or not _NP_AVAILABLE:
        return frame

    display = frame.copy()

    # Run landmark extraction and draw dots
    if extractor is not None:
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            ) as hands:
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    for hand_lms in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            display,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                        )
        except Exception:
            pass

    h, w = display.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top banner: current label
    label_text = f"LABEL: {label}  ({label_idx+1}/{total_labels})"
    cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(display, label_text, (10, 35), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Rep counter
    rep_text = f"Rep {rep}/{total_reps}"
    cv2.putText(display, rep_text, (10, 80), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Phase / countdown indicator
    if phase == "countdown" and countdown is not None:
        big_text = str(countdown)
        text_size = cv2.getTextSize(big_text, font, 5.0, 8)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(display, big_text, (tx, ty), font, 5.0, (0, 255, 255), 8, cv2.LINE_AA)
    elif phase == "recording":
        cv2.putText(display, "RECORDING", (10, h - 20), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        # Blinking red dot
        if int(time.monotonic() * 2) % 2 == 0:
            cv2.circle(display, (w - 30, 25), 12, (0, 0, 255), -1)
    elif phase == "rest":
        cv2.putText(display, "REST", (10, h - 20), font, 1.2, (0, 200, 0), 2, cv2.LINE_AA)

    return display


# ---------------------------------------------------------------------------
# Test-webcam mode
# ---------------------------------------------------------------------------

def run_test_webcam(camera_index: int, duration_s: float = 30.0) -> None:
    """
    Open camera, run MediaPipe, print detected poses to terminal.
    No files are saved.
    """
    if not _CV2_AVAILABLE:
        print("[ERROR] opencv-python is not installed.")
        sys.exit(1)
    if not _MP_AVAILABLE:
        print("[ERROR] mediapipe is not installed.")
        sys.exit(1)

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}.")
        sys.exit(1)

    extractor = HandLandmarkExtractor()
    print(f"Webcam test mode: running for {duration_s:.0f} s. Press Q to quit early.")
    print("Show your hand to the camera. Landmark coordinates will print below.\n")

    deadline = time.monotonic() + duration_s
    frame_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                landmarks = extractor.extract(frame)
                if landmarks is not None:
                    ts = time.monotonic()
                    # Print wrist and fingertip positions (landmarks 0, 4, 8, 12, 16, 20)
                    keypoints = landmarks.reshape(21, 3)
                    tips = {0: "wrist", 4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}
                    parts = "  ".join(
                        f"{name}=({keypoints[i,0]:+.2f},{keypoints[i,1]:+.2f})"
                        for i, name in tips.items()
                    )
                    print(f"  t={ts:.2f}  {parts}")

            cv2.imshow("EMG-ASL Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_count} frames captured.")


# ---------------------------------------------------------------------------
# Core recording session
# ---------------------------------------------------------------------------

def _run_paired_session(args: argparse.Namespace) -> None:
    if not _CV2_AVAILABLE:
        print("[ERROR] opencv-python is required. Run: pip install 'opencv-python>=4.9'")
        sys.exit(1)
    if not _PD_AVAILABLE or not _NP_AVAILABLE:
        print("[ERROR] pandas and numpy are required. Run: pip install pandas numpy")
        sys.exit(1)

    # Resolve label list
    if args.labels:
        invalid = [l for l in args.labels if l not in ASL_LABELS]
        if invalid:
            print(f"[ERROR] Unknown labels: {invalid}  Valid: {ASL_LABELS}")
            sys.exit(1)
        labels = args.labels
    else:
        labels = list(ASL_LETTERS)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{args.participant}_{date_str}"

    emg_path = out_dir / f"{stem}.csv"
    video_path = out_dir / f"{stem}_video.mp4"
    labels_path = out_dir / f"{stem}_labels.csv"

    print("=" * 60)
    print("EMG-ASL Paired Recording Session")
    print("=" * 60)
    print(f"Participant   : {args.participant}")
    print(f"Mode          : {'SYNTHETIC' if args.synthetic else 'HARDWARE BLE'}")
    print(f"Labels        : {labels}")
    print(f"Reps per label: {args.reps}")
    print(f"Hold duration : {args.hold_duration} s")
    print(f"Camera index  : {args.camera_index}")
    print(f"Device        : {args.device_name}")
    print(f"Output EMG    : {emg_path}")
    print(f"Output Video  : {video_path}")
    print(f"Output Labels : {labels_path}")
    print()

    # Shared state
    session_epoch_ms: float = time.monotonic() * 1000.0
    emg_samples: List[EMGSample] = []
    frame_ticks: List[VideoTick] = []
    vision_labels: List[VisionLabel] = []
    current_label_ref: List[str] = [""]  # single-element list used as mutable cell

    start_event = threading.Event()
    stop_event = threading.Event()
    recording_event = threading.Event()

    # Build EMG thread
    if args.synthetic:
        emg_thread: threading.Thread = SyntheticEMGThread(
            labels=labels,
            reps=args.reps,
            hold_duration=args.hold_duration,
            session_epoch_ms=session_epoch_ms,
            emg_samples=emg_samples,
            start_event=start_event,
            stop_event=stop_event,
            recording_event=recording_event,
            current_label_ref=current_label_ref,
        )
    else:
        emg_thread = EMGThread(
            device_name=args.device_name,
            session_epoch_ms=session_epoch_ms,
            emg_samples=emg_samples,
            start_event=start_event,
            stop_event=stop_event,
            recording_event=recording_event,
            current_label_ref=current_label_ref,
        )

    # Build video thread
    no_webcam = getattr(args, "no_webcam", False) and args.synthetic
    if no_webcam:
        video_thread: threading.Thread = BlankVideoThread(
            output_path=video_path,
            session_epoch_ms=session_epoch_ms,
            frame_ticks=frame_ticks,
            start_event=threading.Event(),  # separate event; blank thread starts immediately
            stop_event=stop_event,
        )
        video_start_event = video_thread._start_event  # type: ignore[attr-defined]
    else:
        video_thread = VideoThread(
            camera_index=args.camera_index,
            output_path=video_path,
            session_epoch_ms=session_epoch_ms,
            frame_ticks=frame_ticks,
            vision_labels=vision_labels,
            start_event=threading.Event(),
            stop_event=stop_event,
            recording_event=recording_event,
            current_label_ref=current_label_ref,
        )
        video_start_event = video_thread._start_event  # type: ignore[attr-defined]

    # Start threads
    emg_thread.start()
    video_thread.start()

    # Wait for both to signal ready
    print("Waiting for EMG thread to initialize...")
    start_event.wait(timeout=15.0)
    emg_err = getattr(emg_thread, "error", None)
    if emg_err:
        print(f"\n[ERROR] EMG thread: {emg_err}")
        stop_event.set()
        sys.exit(1)

    print("Waiting for video thread to initialize...")
    video_start_event.wait(timeout=10.0)
    vid_err = getattr(video_thread, "error", None)
    if vid_err:
        print(f"\n[ERROR] Video thread: {vid_err}")
        stop_event.set()
        sys.exit(1)

    # Determine real FPS for display purposes
    vid_fps = getattr(video_thread, "fps", 30.0)
    print(f"[OK] Video thread ready at {vid_fps:.1f} fps\n")

    # Optional MediaPipe extractor for the preview window only (main thread)
    preview_extractor: Optional[HandLandmarkExtractor] = None
    if _MP_AVAILABLE and not no_webcam:
        try:
            preview_extractor = HandLandmarkExtractor()
        except Exception:
            preview_extractor = None

    try:
        for label_idx, label in enumerate(labels):
            print(f"\n{'=' * 40}")
            print(f"LABEL: {label}  ({label_idx + 1}/{len(labels)})")
            print(f"{'=' * 40}")

            for rep in range(1, args.reps + 1):
                print(f"\n  GET READY: {label}  rep {rep}/{args.reps}  -- press ENTER when ready")
                input()

                current_label_ref[0] = label

                # Countdown phase: 3 seconds with live preview
                for tick in range(3, 0, -1):
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        if not no_webcam and _CV2_AVAILABLE:
                            cap_ref = getattr(video_thread, "_cap", None)
                            if cap_ref is not None and cap_ref.isOpened():
                                ok, frame = cap_ref.read()
                                if ok:
                                    display = _draw_overlay(
                                        frame,
                                        label=label,
                                        rep=rep,
                                        total_reps=args.reps,
                                        label_idx=label_idx,
                                        total_labels=len(labels),
                                        phase="countdown",
                                        countdown=tick,
                                        recording=False,
                                        extractor=preview_extractor,
                                    )
                                    cv2.imshow("EMG-ASL Recording", display)
                                    if cv2.waitKey(1) & 0xFF == ord("q"):
                                        raise KeyboardInterrupt
                        time.sleep(0.01)

                # Recording phase
                print(f"  GO! Recording for {args.hold_duration:.1f} s ...", end=" ", flush=True)
                recording_event.set()
                t_start = time.monotonic()

                while (time.monotonic() - t_start) < args.hold_duration:
                    if not no_webcam and _CV2_AVAILABLE:
                        cap_ref = getattr(video_thread, "_cap", None)
                        if cap_ref is not None and cap_ref.isOpened():
                            ok, frame = cap_ref.read()
                            if ok:
                                display = _draw_overlay(
                                    frame,
                                    label=label,
                                    rep=rep,
                                    total_reps=args.reps,
                                    label_idx=label_idx,
                                    total_labels=len(labels),
                                    phase="recording",
                                    countdown=None,
                                    recording=True,
                                    extractor=preview_extractor,
                                )
                                cv2.imshow("EMG-ASL Recording", display)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    raise KeyboardInterrupt
                    time.sleep(0.01)

                recording_event.clear()
                n_emg = sum(1 for s in emg_samples if s[2] == label)
                print(f"done  ({n_emg} EMG samples total for {label})")

                # Rest phase (1.5 s)
                rest_deadline = time.monotonic() + 1.5
                print("  Rest 1.5 s ...", end=" ", flush=True)
                while time.monotonic() < rest_deadline:
                    if not no_webcam and _CV2_AVAILABLE:
                        cap_ref = getattr(video_thread, "_cap", None)
                        if cap_ref is not None and cap_ref.isOpened():
                            ok, frame = cap_ref.read()
                            if ok:
                                display = _draw_overlay(
                                    frame,
                                    label=label,
                                    rep=rep,
                                    total_reps=args.reps,
                                    label_idx=label_idx,
                                    total_labels=len(labels),
                                    phase="rest",
                                    countdown=None,
                                    recording=False,
                                    extractor=None,
                                )
                                cv2.imshow("EMG-ASL Recording", display)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    raise KeyboardInterrupt
                    time.sleep(0.01)
                print("ready.")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving partial data...")

    finally:
        stop_event.set()
        if _CV2_AVAILABLE:
            cv2.destroyAllWindows()
        if preview_extractor is not None:
            preview_extractor.close()
        emg_thread.join(timeout=5.0)
        video_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    if not emg_samples:
        print("\n[WARNING] No EMG samples were collected. No files written.")
        return

    print(f"\nSaving EMG CSV to {emg_path} ...")
    _write_emg_csv(emg_path, emg_samples)

    print(f"Saving vision labels CSV to {labels_path} ...")
    _write_labels_csv(labels_path, vision_labels)

    # ------------------------------------------------------------------
    # Auto-label step: merge vision labels into EMG CSV
    # ------------------------------------------------------------------
    print("\nRunning auto-label sync (sync_labels_to_emg) ...")
    import pandas as pd

    emg_df = pd.read_csv(emg_path)
    labels_csv_df = pd.read_csv(labels_path)

    if labels_csv_df.empty:
        print("  [INFO] No vision labels found. EMG file has no auto-labels.")
        merged_df = emg_df
    else:
        merged_df = sync_labels_to_emg(labels_csv_df, emg_df, tolerance_ms=50.0)
        labeled_count = (merged_df.get("label", pd.Series(dtype=str)) != "UNLABELED").sum()
        total_count = len(merged_df)
        print(f"  {labeled_count}/{total_count} EMG rows received a vision label "
              f"({100.0 * labeled_count / max(total_count, 1):.1f}% coverage)")

    # Preview first 10 rows
    print("\nMerged data preview (first 10 rows):")
    print(merged_df.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    from collections import Counter

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    label_counts = Counter(s[2] for s in emg_samples)
    for lbl in labels:
        count = label_counts.get(lbl, 0)
        expected = args.reps * int(args.hold_duration * SAMPLE_RATE)
        print(f"  {lbl:>12s} : {count:5d} EMG samples  (expected ~{expected})")

    total_emg = sum(label_counts.values())
    total_frames = len(frame_ticks)
    total_vision = len(vision_labels)

    print(f"\n  Total EMG samples   : {total_emg}")
    print(f"  Total video frames  : {total_frames}  ({vid_fps:.1f} fps)")
    print(f"  Total vision labels : {total_vision}")
    print(f"\n  EMG CSV   : {emg_path}")
    print(f"  Video MP4 : {video_path}")
    print(f"  Labels CSV: {labels_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Argument parsing and main entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record simultaneous webcam video + EMG for cross-modal training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hardware recording (all letters, 5 reps each):
  python scripts/record_paired_session.py --participant P001 --reps 5

  # Hardware recording (subset of labels):
  python scripts/record_paired_session.py --participant P001 --labels A B C --reps 5

  # Synthetic mode (no BLE hardware needed):
  python scripts/record_paired_session.py --participant TEST001 --synthetic --labels A B C --reps 3

  # Synthetic mode without camera:
  python scripts/record_paired_session.py --participant TEST001 --synthetic --no-webcam --labels A B --reps 2

  # Live webcam + MediaPipe test (no files saved):
  python scripts/record_paired_session.py --test-webcam
        """,
    )

    parser.add_argument(
        "--test-webcam",
        action="store_true",
        dest="test_webcam",
        help="Open camera and run MediaPipe for 30 s. No files saved.",
    )
    parser.add_argument(
        "--participant",
        help="Participant ID (e.g. P001). Used in output filenames.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="ASL label strings to record. Defaults to A-Z.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Repetitions per label (default: 5).",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=3.0,
        dest="hold_duration",
        help="Recording duration per rep in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/",
        dest="output_dir",
        help="Output directory (default: data/raw/).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic EMG data instead of BLE hardware.",
    )
    parser.add_argument(
        "--no-webcam",
        action="store_true",
        dest="no_webcam",
        help="Write blank video frames (only valid with --synthetic).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        dest="camera_index",
        help="OpenCV camera device index (default: 0).",
    )
    parser.add_argument(
        "--device-name",
        default=BLE_DEVICE_NAME,
        dest="device_name",
        help=f"BLE device name (default: {BLE_DEVICE_NAME}).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.test_webcam:
        run_test_webcam(camera_index=args.camera_index, duration_s=30.0)
        return

    if not args.participant:
        parser.error("--participant is required unless --test-webcam is set.")

    if args.no_webcam and not args.synthetic:
        parser.error("--no-webcam can only be used together with --synthetic.")

    _run_paired_session(args)


if __name__ == "__main__":
    main()

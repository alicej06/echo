"""
Vision-based teacher for EMG data labeling.

Uses MediaPipe Hands to extract hand pose landmarks from video,
then trains a simple SVC pose classifier that can auto-label
synchronized EMG recordings.

Pipeline:
  Video frames → MediaPipe → 63-dim landmarks → SVC → ASL label
  Synchronized EMG + labels → EMG classifier training data
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np

from src.constants import ASL_CLASSES, N_CLASSES, SAMPLE_RATE


class HandLandmarkExtractor:
    """Extract wrist-relative, L2-normalized hand landmarks from frames."""

    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            raise ImportError("mediapipe not installed. Run: pip install mediapipe")

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 63-dim landmark vector from a BGR frame.
        Returns None if no hand detected.
        """
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0].landmark
        coords = np.array([[p.x, p.y, p.z] for p in lm])  # (21, 3)
        # Wrist-relative normalization
        coords -= coords[0:1]
        # L2 normalize
        norm = np.linalg.norm(coords) + 1e-8
        return (coords / norm).flatten().astype(np.float32)  # (63,)

    def __del__(self):
        if hasattr(self, "hands"):
            self.hands.close()


class SimpleASLPoseClassifier:
    """Sklearn SVC pipeline for ASL pose classification."""

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ])
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleASLPoseClassifier":
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: str):
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SimpleASLPoseClassifier":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class VisionTeacher:
    """
    Auto-label EMG data using synchronized webcam footage.
    """

    def __init__(self, classifier: SimpleASLPoseClassifier, confidence_threshold: float = 0.7):
        self.classifier = classifier
        self.extractor = HandLandmarkExtractor()
        self.threshold = confidence_threshold

    def stream_labels(self, cap, fps: float = 30.0) -> Generator[Tuple[float, Optional[int], float], None, None]:
        """
        Yield (timestamp, class_idx, confidence) from a cv2 VideoCapture.
        timestamp is in seconds from capture start.
        """
        import cv2
        frame_idx = 0
        start = time.monotonic()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps
            landmarks = self.extractor.extract(frame)
            if landmarks is not None:
                probs = self.classifier.predict_proba(landmarks[None])[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                if conf >= self.threshold:
                    yield ts, pred_idx, conf
                else:
                    yield ts, None, conf
            else:
                yield ts, None, 0.0
            frame_idx += 1

    def label_video_file(self, video_path: str, fps: float = 30.0) -> list:
        """Label all frames in a video file. Returns list of (ts, class_idx, conf)."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        results = list(self.stream_labels(cap, fps=actual_fps))
        cap.release()
        return results


def sync_labels_to_emg(
    labels: list,
    emg_timestamps: np.ndarray,
    tolerance_s: float = 0.05,
) -> np.ndarray:
    """
    Nearest-neighbor sync: assign vision label to each EMG sample.

    Args:
        labels: [(timestamp, class_idx, confidence), ...]
        emg_timestamps: (N,) array of EMG sample timestamps in seconds
        tolerance_s: max allowed time gap (default 50ms)

    Returns:
        label_array: (N,) int32, -1 for unmatched samples
    """
    label_array = np.full(len(emg_timestamps), -1, dtype=np.int32)
    if not labels:
        return label_array

    vision_ts = np.array([l[0] for l in labels])
    vision_cls = np.array([l[1] if l[1] is not None else -1 for l in labels])

    for i, ts in enumerate(emg_timestamps):
        idx = np.argmin(np.abs(vision_ts - ts))
        if np.abs(vision_ts[idx] - ts) <= tolerance_s and vision_cls[idx] >= 0:
            label_array[i] = vision_cls[idx]

    return label_array

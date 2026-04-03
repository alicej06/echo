"""
Real-time EMG-ASL pipeline demo — no server, no hardware required.

Simulates 5 seconds of synthetic EMG data for a chosen label, slides a
200 ms window with 50% overlap over it, and runs the full
  filter -> feature extraction -> LSTM inference
chain for every window, reporting prediction and per-step latency.

Usage
-----
    python scripts/demo_pipeline.py               # defaults to label HELLO
    python scripts/demo_pipeline.py --label A
    python scripts/demo_pipeline.py --label THANK_YOU
    python scripts/demo_pipeline.py --duration 10

If ``models/asl_emg_classifier.pt`` exists relative to the repo root it
will be loaded and used; otherwise a fresh random-weight model is created
and each prediction is annotated with ``[random weights]``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repository root on sys.path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants (mirrors src/utils/constants.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 200           # Hz
WINDOW_SIZE_SAMPLES = 40    # 200 ms at 200 Hz
STEP_SIZE_SAMPLES = 20      # 50% overlap -> 100 ms step
N_CHANNELS = 8
FEATURE_SIZE = 80           # 8 ch * 10 features
INPUT_SIZE = 320            # train_model flattens (40, 8) -> 320; model input dim
NUM_CLASSES = 36

ASL_LABELS: list[str] = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + ["HELLO", "THANK_YOU", "PLEASE", "YES", "NO", "HELP", "WATER", "MORE", "STOP", "GO"]
)

PYTORCH_MODEL_PATH = REPO_ROOT / "models" / "asl_emg_classifier.pt"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: simulate real-time EMG-ASL inference pipeline"
    )
    parser.add_argument(
        "--label",
        default="HELLO",
        choices=ASL_LABELS,
        help="ASL label to simulate (default: HELLO)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulated recording duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Synthetic EMG generator
# ---------------------------------------------------------------------------


def generate_synthetic_emg(
    duration_s: float,
    n_channels: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a (n_samples, n_channels) array of synthetic EMG-like data.

    The signal is bandlimited Gaussian noise with a mild rectified burst
    pattern to loosely resemble a muscle contraction gesture.

    Parameters
    ----------
    duration_s:
        Length of the recording in seconds.
    n_channels:
        Number of EMG channels.
    sample_rate:
        Samples per second.
    rng:
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_samples, n_channels), dtype float64.
    """
    n_samples = int(duration_s * sample_rate)

    # Base broadband noise in a realistic ADC range (~±2048 for 12-bit)
    noise = rng.normal(loc=0.0, scale=300.0, size=(n_samples, n_channels))

    # Simulate muscle bursts: amplitude envelope oscillates between 0.4 and 1.0
    t = np.linspace(0.0, duration_s, n_samples)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz burst rhythm
    signal = noise * envelope[:, np.newaxis]

    # Add small per-channel DC offset to exercise the DC-removal filter step
    dc_offset = rng.uniform(-50.0, 50.0, size=(1, n_channels))
    return signal + dc_offset


# ---------------------------------------------------------------------------
# Model loading / creation
# ---------------------------------------------------------------------------


def load_or_create_model(
    model_path: Path,
) -> tuple:
    """Return (model, using_real_weights: bool).

    Tries to load a saved checkpoint.  Falls back to random weights if the
    file does not exist.
    """
    from src.models.lstm_classifier import ASLEMGClassifier

    if model_path.exists():
        try:
            model = ASLEMGClassifier.load(model_path)
            print(f"[Demo] Loaded trained model from {model_path}")
            return model, True
        except Exception as exc:  # noqa: BLE001
            print(f"[Demo] WARNING: could not load {model_path}: {exc}")
            print("[Demo] Falling back to random-weight model.")

    # Fresh model with random weights
    model = ASLEMGClassifier(
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        label_names=ASL_LABELS,
    )
    model.eval()
    return model, False


# ---------------------------------------------------------------------------
# Per-window inference
# ---------------------------------------------------------------------------


def infer_window(
    model,
    window: np.ndarray,
) -> tuple[str, float, float]:
    """Filter, extract features, and run model inference for one window.

    Parameters
    ----------
    model:
        ASLEMGClassifier instance.
    window:
        Raw EMG window, shape (WINDOW_SIZE_SAMPLES, N_CHANNELS).

    Returns
    -------
    (label, confidence, latency_ms)
    """
    from src.utils.filters import apply_full_filter_chain
    from src.utils.features import extract_features

    t0 = time.perf_counter()

    # Step 1: filter (DC remove -> bandpass -> notch -> RMS normalise)
    filtered = apply_full_filter_chain(
        window,
        fs=float(SAMPLE_RATE),
        lowcut=20.0,
        highcut=90.0,   # Nyquist for 200 Hz fs is 100 Hz; stay below
        notch_freq=60.0,
    )

    # Step 2: feature extraction -> (80,) for 8 channels
    features_80 = extract_features(filtered, fs=float(SAMPLE_RATE))

    # Step 3: the trained model was built with input_size=320 (raw flattened
    # windows).  We pad/tile the 80-dim feature vector to 320-dim so the
    # demo works with the same model architecture used during training.
    # In production, either:
    #   (a) train with feature vectors (input_size=80), or
    #   (b) pass the raw flattened window directly (shape (320,)).
    # For this demo we tile 4x so the vector length matches INPUT_SIZE=320.
    if features_80.shape[0] != INPUT_SIZE:
        repeats = INPUT_SIZE // features_80.shape[0]
        remainder = INPUT_SIZE % features_80.shape[0]
        features_input = np.concatenate(
            [np.tile(features_80, repeats), features_80[:remainder]]
        )
    else:
        features_input = features_80

    # Step 4: model inference
    label, confidence = model.predict(features_input)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return label, confidence, latency_ms


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print()
    print(f"[Demo] Generating {args.duration:.0f}s of synthetic EMG for label: {args.label}")
    print(f"[Demo] Sample rate: {SAMPLE_RATE} Hz | Window: {WINDOW_SIZE_SAMPLES} samples "
          f"({WINDOW_SIZE_SAMPLES * 1000 // SAMPLE_RATE} ms) | "
          f"Step: {STEP_SIZE_SAMPLES} samples ({STEP_SIZE_SAMPLES * 1000 // SAMPLE_RATE} ms overlap)")

    # Generate raw signal
    signal = generate_synthetic_emg(
        duration_s=args.duration,
        n_channels=N_CHANNELS,
        sample_rate=SAMPLE_RATE,
        rng=rng,
    )
    n_samples = signal.shape[0]

    # Compute window count
    n_windows = (n_samples - WINDOW_SIZE_SAMPLES) // STEP_SIZE_SAMPLES + 1
    print(f"[Demo] Total samples: {n_samples} | Windows to process: {n_windows}")
    print()

    # Load or create model
    model, using_real_weights = load_or_create_model(PYTORCH_MODEL_PATH)
    weight_tag = "" if using_real_weights else "  [random weights]"
    if not using_real_weights:
        print("[Demo] NOTE: no trained checkpoint found — using random weights.")
        print(f"[Demo] Train and save to: {PYTORCH_MODEL_PATH}")
        print()

    latencies: list[float] = []

    print(f"[Demo] Processing {n_windows} windows...")
    print()

    for i in range(n_windows):
        start = i * STEP_SIZE_SAMPLES
        end = start + WINDOW_SIZE_SAMPLES
        window = signal[start:end]  # (40, 8)

        try:
            label, confidence, latency_ms = infer_window(model, window)
        except Exception as exc:  # noqa: BLE001
            print(f"Window {i + 1:3d}/{n_windows} | ERROR: {exc}")
            continue

        latencies.append(latency_ms)

        print(
            f"Window {i + 1:3d}/{n_windows} | "
            f"Predicted: {label:<9s} (conf={confidence:.2f}) | "
            f"Latency: {latency_ms:.1f}ms"
            f"{weight_tag}"
        )

    print()
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(
            f"[Demo] Done. Processed {len(latencies)}/{n_windows} windows successfully."
        )
        print(
            f"[Demo] Inference latency — "
            f"avg: {avg_latency:.1f}ms | "
            f"min: {min_latency:.1f}ms | "
            f"max: {max_latency:.1f}ms"
        )
    else:
        print("[Demo] No windows were processed successfully.")

    if not using_real_weights:
        print()
        print("[Demo] Predictions above are meaningless (random weights).")
        print(f"[Demo] Train a model and save it to {PYTORCH_MODEL_PATH} for real results.")


if __name__ == "__main__":
    main()

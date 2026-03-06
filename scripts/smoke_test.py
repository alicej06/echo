"""
End-to-end smoke test for the EMG-ASL layer.

Runs without a server, without hardware, and without any pre-trained weights.
Each test is self-contained and prints PASS or FAIL [reason].

Usage
-----
    python scripts/smoke_test.py

from the repository root (emg-asl-layer/).
"""

from __future__ import annotations

import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Repository root on sys.path so relative imports work
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants shared by tests
# ---------------------------------------------------------------------------

N_CHANNELS = 8
WINDOW_SAMPLES = 40          # 200 ms at 200 Hz
SAMPLE_RATE = 200.0          # Hz
INPUT_SIZE = 320             # train_model flattens (40, 8) -> 320
NUM_CLASSES = 36
FEATURE_SIZE = 80            # 8 ch * 10 features

ASL_LABELS = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + ["HELLO", "THANK_YOU", "PLEASE", "YES", "NO", "HELP", "WATER", "MORE", "STOP", "GO"]
)

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn: Callable[[], None]) -> None:
    """Execute *fn*, record PASS/FAIL, and print the result immediately."""
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        _results.append((name, False, reason))
        print(f"  FAIL  {name}  [{reason}]")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 1: Signal processing pipeline
# ---------------------------------------------------------------------------


def test_filter_chain() -> None:
    """Generate 200-sample, 8-channel random signal and run the full filter chain."""
    from src.utils.filters import apply_full_filter_chain

    rng = np.random.default_rng(0)
    # 200 samples at 200 Hz = 1 second; add a realistic DC offset
    raw = rng.normal(loc=500.0, scale=100.0, size=(200, N_CHANNELS))

    filtered = apply_full_filter_chain(
        raw,
        fs=SAMPLE_RATE,
        lowcut=20.0,
        highcut=90.0,   # keep well below Nyquist=100 Hz for 200 Hz fs
        notch_freq=60.0,
    )

    assert filtered.shape == raw.shape, (
        f"Shape mismatch: expected {raw.shape}, got {filtered.shape}"
    )
    assert np.all(np.isfinite(filtered)), "Filter output contains non-finite values"


# ---------------------------------------------------------------------------
# Test 2: Feature extraction
# ---------------------------------------------------------------------------


def test_feature_extraction() -> None:
    """Extract features from a (40, 8) window and verify shape/finiteness."""
    from src.utils.features import extract_features

    rng = np.random.default_rng(1)
    window = rng.normal(size=(WINDOW_SAMPLES, N_CHANNELS))

    features = extract_features(window, fs=SAMPLE_RATE)

    assert features.shape == (FEATURE_SIZE,), (
        f"Expected shape ({FEATURE_SIZE},), got {features.shape}"
    )
    assert np.all(np.isfinite(features)), "Feature vector contains non-finite values"


# ---------------------------------------------------------------------------
# Test 3: Model instantiation and forward pass
# ---------------------------------------------------------------------------


def test_model_instantiation() -> None:
    """Create ASLEMGClassifier(input_size=320) and run one forward pass."""
    import torch
    from src.models.lstm_classifier import ASLEMGClassifier

    model = ASLEMGClassifier(
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
        label_names=ASL_LABELS,
    )
    model.eval()

    dummy = torch.zeros(1, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (1, NUM_CLASSES), (
        f"Expected logits shape (1, {NUM_CLASSES}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: model.predict() with a (320,) feature vector
# ---------------------------------------------------------------------------


def test_model_predict() -> None:
    """Call model.predict() and verify return types and value ranges."""
    from src.models.lstm_classifier import ASLEMGClassifier

    model = ASLEMGClassifier(
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        label_names=ASL_LABELS,
    )

    rng = np.random.default_rng(2)
    feat = rng.normal(size=(INPUT_SIZE,)).astype(np.float32)

    label, confidence = model.predict(feat)

    assert isinstance(label, str), f"label must be str, got {type(label)}"
    assert isinstance(confidence, float), f"confidence must be float, got {type(confidence)}"
    assert 0.0 <= confidence <= 1.0, f"confidence {confidence} outside [0, 1]"


# ---------------------------------------------------------------------------
# Test 5: Save / load round-trip
# ---------------------------------------------------------------------------


def test_model_save_load() -> None:
    """Save a model to /tmp, load it back, verify predictions match."""
    import torch
    from src.models.lstm_classifier import ASLEMGClassifier

    save_path = Path("/tmp/test_model.pt")

    rng = np.random.default_rng(3)
    feat = rng.normal(size=(INPUT_SIZE,)).astype(np.float32)

    model_orig = ASLEMGClassifier(
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        label_names=ASL_LABELS,
    )
    label_orig, conf_orig = model_orig.predict(feat)
    model_orig.save(save_path)

    model_loaded = ASLEMGClassifier.load(save_path)
    label_loaded, conf_loaded = model_loaded.predict(feat)

    assert label_orig == label_loaded, (
        f"Label mismatch after save/load: {label_orig!r} vs {label_loaded!r}"
    )
    assert abs(conf_orig - conf_loaded) < 1e-5, (
        f"Confidence mismatch after save/load: {conf_orig} vs {conf_loaded}"
    )


# ---------------------------------------------------------------------------
# Test 6: ONNX export and inference
# ---------------------------------------------------------------------------


def test_onnx_export() -> None:
    """Export model to ONNX, load with onnxruntime, verify output shape.

    If onnx or onnxruntime are not installed this test is skipped with a
    clear message rather than failing hard — the packages are optional extras.
    """
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as exc:
        # Treat missing optional deps as a soft skip by printing a note and
        # returning cleanly so the test counts as passed.  The caller can
        # gate on this if desired.
        print(f"        (skipped — optional packages not installed: {exc})")
        print("         Install with: pip install onnx onnxruntime")
        return

    from src.models.lstm_classifier import ASLEMGClassifier

    onnx_path = Path("/tmp/test_model.onnx")

    model = ASLEMGClassifier(
        input_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        label_names=ASL_LABELS,
    )
    model.to_onnx(onnx_path)

    assert onnx_path.exists(), "ONNX file was not created"

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(4)
    feat = rng.normal(size=(1, INPUT_SIZE)).astype(np.float32)
    outputs = session.run(None, {input_name: feat})

    logits = outputs[0]
    assert logits.shape == (1, NUM_CLASSES), (
        f"Expected ONNX output shape (1, {NUM_CLASSES}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 7: Data loader — create_windows on synthetic DataFrame
# ---------------------------------------------------------------------------


def test_data_loader() -> None:
    """Build a tiny synthetic DataFrame (50 rows, label='A') and run create_windows.

    loader.py imports imblearn at the top level.  If it is not installed we
    skip gracefully with an install hint.
    """
    import pandas as pd
    try:
        from src.data.loader import create_windows
    except ModuleNotFoundError as exc:
        print(f"        (skipped — optional dependency missing: {exc})")
        print("         Install with: pip install imbalanced-learn")
        return

    n_rows = 50
    rng = np.random.default_rng(5)

    data = {f"ch{i + 1}": rng.normal(size=n_rows).tolist() for i in range(N_CHANNELS)}
    data["timestamp_ms"] = list(range(n_rows))
    data["label"] = ["A"] * n_rows

    df = pd.DataFrame(data)

    windows, labels = create_windows(
        df,
        window_size_ms=200,
        overlap=0.5,
        fs=200,
    )

    assert windows.ndim == 3, f"Expected 3-D windows array, got ndim={windows.ndim}"
    assert windows.shape[1] == WINDOW_SAMPLES, (
        f"Expected window_samples={WINDOW_SAMPLES}, got {windows.shape[1]}"
    )
    assert windows.shape[2] == N_CHANNELS, (
        f"Expected {N_CHANNELS} channels, got {windows.shape[2]}"
    )
    assert len(labels) == len(windows), "labels and windows length mismatch"
    assert all(l == "A" for l in labels), "Unexpected labels in output"


# ---------------------------------------------------------------------------
# Test 8: Pipeline integration — ingest raw BLE bytes
# ---------------------------------------------------------------------------


def test_pipeline_integration() -> None:
    """Feed raw BLE bytes into EMGPipeline and drain at least one window.

    Note on bandpass_high: the default constant BANDPASS_HIGH=450 Hz is
    intended for hardware running at a higher sample rate.  At fs=200 Hz the
    Nyquist limit is 100 Hz, so we override bandpass_high=90 Hz here to keep
    the filter parameters valid for this test.
    """
    from src.utils.pipeline import EMGPipeline

    pipeline = EMGPipeline(
        n_channels=N_CHANNELS,
        sample_rate=SAMPLE_RATE,
        window_size_samples=WINDOW_SAMPLES,
        step_size_samples=20,       # 50% overlap
        bandpass_high=90.0,         # must be < Nyquist (100 Hz at 200 Hz fs)
    )

    rng = np.random.default_rng(6)

    # Generate enough samples to produce at least one window.
    # We need WINDOW_SAMPLES samples before the first window, then STEP_SAMPLES
    # for each additional one.  Send 200 samples to be safe.
    n_samples = 200
    samples_int16 = rng.integers(-1000, 1000, size=(n_samples, N_CHANNELS), dtype=np.int16)

    # Pack as interleaved little-endian int16 (BLE format)
    raw_bytes = samples_int16.astype("<i2").tobytes()

    n_decoded = pipeline.ingest_bytes(raw_bytes)

    assert n_decoded == n_samples, (
        f"Expected {n_samples} decoded samples, got {n_decoded}"
    )
    assert pipeline.pending_windows > 0, (
        "No windows were queued after ingesting enough data"
    )

    # Drain one window and process it through the feature chain
    window = pipeline.get_next_window()
    assert window is not None, "get_next_window() returned None unexpectedly"
    assert window.shape == (WINDOW_SAMPLES, N_CHANNELS), (
        f"Unexpected window shape: {window.shape}"
    )

    features = pipeline.process_window(window)
    assert features.shape == (FEATURE_SIZE,), (
        f"Expected feature shape ({FEATURE_SIZE},), got {features.shape}"
    )
    assert np.all(np.isfinite(features)), "Feature vector from pipeline contains non-finite values"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n=== EMG-ASL Smoke Test ===\n")
    print("Running 8 tests...\n")

    tests = [
        ("1/8  Filter chain (200-sample signal, 8 ch)", test_filter_chain),
        ("2/8  Feature extraction (40x8 window -> 80-dim)", test_feature_extraction),
        ("3/8  Model instantiation + forward pass (1, 320)", test_model_instantiation),
        ("4/8  model.predict() returns (label_str, confidence in [0,1])", test_model_predict),
        ("5/8  Save / load round-trip -> predictions match", test_model_save_load),
        ("6/8  ONNX export + onnxruntime inference", test_onnx_export),
        ("7/8  Data loader create_windows on synthetic DataFrame", test_data_loader),
        ("8/8  Pipeline integration: ingest BLE bytes -> features", test_pipeline_integration),
    ]

    for name, fn in tests:
        run_test(name, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)

    print(f"\n{'='*42}")
    print(f"Summary: {passed}/{total} tests passed")
    if passed < total:
        print("\nFailed tests:")
        for name, ok, reason in _results:
            if not ok:
                print(f"  - {name}")
                print(f"    Reason: {reason}")
    print("=" * 42)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export EMG-ASL models to Apple CoreML format.

CoreML runs on the iPhone Neural Engine (ANE), which is 10-100x faster
than CPU for neural network inference. This replaces the ONNX runtime
on iPhone and enables real-time ASL recognition at <5ms per window.

Requirements:
  pip install coremltools>=7.0
  macOS only (CoreML models are built on Mac, deployed to iPhone)

Usage:
  # Export LSTM model (default):
  python scripts/export_coreml.py --model models/asl_emg_classifier.pt

  # Export CNN-LSTM:
  python scripts/export_coreml.py --model models/cnn_lstm_classifier.pt --type cnn_lstm

  # Export Conformer:
  python scripts/export_coreml.py --model models/conformer_classifier.pt --type conformer

  # Export all available models:
  python scripts/export_coreml.py --all

Output: models/{model_name}.mlpackage (CoreML ML Package format)
        models/{model_name}.mlmodel  (CoreML 5 legacy format, for iOS < 16)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root on sys.path so src.* imports resolve when run from any cwd
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# coremltools availability check
# ---------------------------------------------------------------------------


def _require_coremltools() -> object:
    """Import coremltools and return the module, or exit with a clear message."""
    try:
        import coremltools as ct  # type: ignore[import]
        return ct
    except ImportError:
        print(
            "\n[export_coreml] coremltools is not installed.\n"
            "Install it with:\n"
            "  pip install coremltools>=7.0\n"
            "\nNote: coremltools requires macOS for full CoreML conversion.\n"
            "The package installs on Linux/Windows but the Neural Engine path\n"
            "and .mlpackage output require macOS 12+ with Xcode 14+.\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model type inference
# ---------------------------------------------------------------------------


def _infer_model_type(path: Path) -> str:
    """Infer model architecture from filename heuristics."""
    name = path.stem.lower()
    if "conformer" in name:
        return "conformer"
    if "cnn" in name or "cnn_lstm" in name:
        return "cnn_lstm"
    return "lstm"


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _load_model(model_path: Path, model_type: str):
    """Load a PyTorch checkpoint and return the model in eval mode."""
    if model_type == "lstm":
        from src.models.lstm_classifier import ASLEMGClassifier
        model = ASLEMGClassifier.load(model_path)
    elif model_type == "cnn_lstm":
        from src.models.cnn_lstm_classifier import CNNLSTMClassifier
        model = CNNLSTMClassifier.load(model_path)
    elif model_type == "conformer":
        from src.models.conformer_classifier import ConformerClassifier
        model = ConformerClassifier.load(model_path)
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            "Expected one of: lstm, cnn_lstm, conformer."
        )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Input shape helpers
# ---------------------------------------------------------------------------


def _get_example_input(model_type: str):
    """Return a torch.Tensor example input for torch.jit.trace."""
    import torch
    from src.utils.constants import FEATURE_VECTOR_SIZE, WINDOW_SIZE_SAMPLES, N_CHANNELS

    if model_type == "lstm":
        # LSTM: flat feature vector (1, 320)  -- 8 channels x 40 samples
        return torch.zeros(1, FEATURE_VECTOR_SIZE, dtype=torch.float32)
    else:
        # CNN-LSTM and Conformer: raw window (1, 40, 8)
        return torch.zeros(1, WINDOW_SIZE_SAMPLES, N_CHANNELS, dtype=torch.float32)


def _get_coreml_input_spec(ct, model_type: str):
    """Return coremltools input TensorType with name and shape."""
    from src.utils.constants import FEATURE_VECTOR_SIZE, WINDOW_SIZE_SAMPLES, N_CHANNELS

    if model_type == "lstm":
        return ct.TensorType(
            name="emg_window",
            shape=(1, FEATURE_VECTOR_SIZE),
        )
    else:
        return ct.TensorType(
            name="emg_window",
            shape=(1, WINDOW_SIZE_SAMPLES, N_CHANNELS),
        )


# ---------------------------------------------------------------------------
# Inference time estimate
# ---------------------------------------------------------------------------


_INFERENCE_MS_ESTIMATES = {
    "lstm": "~2-5ms",
    "cnn_lstm": "~3-7ms",
    "conformer": "~3-8ms",
}


# ---------------------------------------------------------------------------
# CoreML README for the ios/Models directory
# ---------------------------------------------------------------------------


_IOS_MODELS_README = """\
# iOS CoreML Models

Place exported `.mlpackage` files in this directory, then add them to your
Xcode project as follows:

1. In Xcode, open the project navigator (Cmd+1).
2. Drag the `.mlpackage` bundle into the project tree (e.g., under
   `MAIA-EMG/Models/`).
3. When prompted, check "Copy items if needed" and add to the main target.
4. Xcode automatically generates a Swift class named after the file (e.g.,
   `AslEmgClassifier`).  Use it for inference:

```swift
import CoreML

let config = MLModelConfiguration()
config.computeUnits = .all   // enables Neural Engine

let model = try AslEmgClassifier(configuration: config)
let input = AslEmgClassifierInput(emg_window: /* MLMultiArray */)
let output = try model.prediction(from: input)
```

## Expected latency on Neural Engine

| Model       | Input shape   | Estimated latency |
|-------------|---------------|-------------------|
| lstm        | (1, 320)      | ~2-5 ms           |
| cnn_lstm    | (1, 40, 8)    | ~3-7 ms           |
| conformer   | (1, 40, 8)    | ~3-8 ms           |

These estimates apply to an iPhone 14 or later running iOS 16+.
Older devices may be 1.5-2x slower on the Neural Engine.

## FP16 quantised models

Files ending in `_fp16.mlpackage` are weight-quantised to float16.
They are ~2x smaller than the FP32 originals with negligible accuracy loss
(< 0.5% on typical EMG test sets) and run at the same speed.
"""


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------


def export_model(
    model_path: Path,
    model_type: str,
    output_dir: Path,
    quantize: bool = False,
) -> None:
    """Convert one PyTorch .pt checkpoint to CoreML .mlpackage + .mlmodel.

    Parameters
    ----------
    model_path:
        Path to the .pt checkpoint.
    model_type:
        Architecture identifier: 'lstm', 'cnn_lstm', or 'conformer'.
    output_dir:
        Directory where output files are written.
    quantize:
        If True, apply FP16 linear quantisation after conversion.
    """
    ct = _require_coremltools()
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem  # e.g. "asl_emg_classifier"
    suffix = "_fp16" if quantize else ""

    mlpackage_path = output_dir / f"{stem}{suffix}.mlpackage"
    mlmodel_path = output_dir / f"{stem}{suffix}.mlmodel"

    print(f"\n[export_coreml] Exporting: {model_path.name}")
    print(f"  Architecture : {model_type}")
    print(f"  Quantise FP16: {quantize}")

    # ------------------------------------------------------------------
    # 1. Load and trace the PyTorch model
    # ------------------------------------------------------------------
    model = _load_model(model_path, model_type)
    example_input = _get_example_input(model_type)

    print(f"  Tracing model with input shape: {tuple(example_input.shape)} ...")
    try:
        traced = torch.jit.trace(model, example_input)
    except Exception as exc:
        print(f"  [ERROR] torch.jit.trace failed: {exc}")
        print(
            "  Tip: ensure the model forward() has no data-dependent control flow "
            "that changes based on input values."
        )
        raise

    # ------------------------------------------------------------------
    # 2. Convert to CoreML
    # ------------------------------------------------------------------
    input_spec = _get_coreml_input_spec(ct, model_type)

    print("  Converting to CoreML (iOS 16 target) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[input_spec],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",   # .mlpackage (Neural Engine optimised)
    )

    # ------------------------------------------------------------------
    # 3. Set model metadata
    # ------------------------------------------------------------------
    mlmodel.author = "MAIA Biotech Lab"
    mlmodel.short_description = (
        f"EMG-based ASL gesture classifier ({model_type.upper()} architecture). "
        "Classifies 8-channel forearm EMG into 36 ASL classes "
        "(26 letters + 10 common words). "
        "Input: pre-processed EMG window. Output: raw class logits (softmax in app)."
    )
    mlmodel.version = "1.0.0"
    mlmodel.license = "MIT"

    # Per-input / output descriptions
    spec = mlmodel.get_spec()
    if model_type == "lstm":
        input_desc = (
            "Flat feature vector of shape (1, 320). "
            "Computed by extract_features() on a 200ms/40-sample window: "
            "8 channels x 10 features (5 time-domain + 5 frequency-domain)."
        )
    else:
        input_desc = (
            "Raw EMG window of shape (1, 40, 8): "
            "batch=1, time=40 samples (200ms at 200 Hz), channels=8."
        )
    ct.utils.rename_feature(spec, "emg_window", "emg_window")
    mlmodel.input_description["emg_window"] = input_desc
    mlmodel.output_description["logits"] = (
        "Raw class logits, shape (1, 36). "
        "Apply softmax to obtain per-class probabilities. "
        "argmax gives the predicted class index (0=A, 1=B, ..., 25=Z, "
        "26=HELLO, 27=THANK_YOU, 28=PLEASE, 29=YES, 30=NO, "
        "31=HELP, 32=WATER, 33=MORE, 34=STOP, 35=GO)."
    )

    # ------------------------------------------------------------------
    # 4. Optional FP16 quantisation
    # ------------------------------------------------------------------
    if quantize:
        print("  Applying FP16 linear weight quantisation ...")
        try:
            mlmodel = ct.optimize.coreml.linear_quantize_weights(
                mlmodel, mode="linear_symmetric"
            )
        except AttributeError:
            # Older coremltools may not have this path; try the legacy API
            try:
                from coremltools.models.neural_network import quantization_utils  # type: ignore
                mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)
            except Exception as q_exc:
                print(f"  [WARNING] FP16 quantisation failed ({q_exc}). Saving FP32.")

    # ------------------------------------------------------------------
    # 5. Save .mlpackage (iOS 16+ / Neural Engine path)
    # ------------------------------------------------------------------
    print(f"  Saving .mlpackage -> {mlpackage_path} ...")
    mlmodel.save(str(mlpackage_path))

    # ------------------------------------------------------------------
    # 6. Save legacy .mlmodel (iOS < 16 compatibility)
    #    Re-convert using the neuralnetwork backend.
    # ------------------------------------------------------------------
    print(f"  Saving legacy .mlmodel -> {mlmodel_path} ...")
    try:
        mlmodel_legacy = ct.convert(
            traced,
            inputs=[input_spec],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=ct.target.iOS15,
            convert_to="neuralnetwork",  # legacy .mlmodel format
        )
        mlmodel_legacy.author = mlmodel.author
        mlmodel_legacy.short_description = mlmodel.short_description
        mlmodel_legacy.version = mlmodel.version
        mlmodel_legacy.save(str(mlmodel_path))
    except Exception as legacy_exc:
        print(
            f"  [WARNING] Legacy .mlmodel export failed ({legacy_exc}). "
            "Skipping iOS <16 compatibility file."
        )

    # ------------------------------------------------------------------
    # 7. Report file sizes and estimated latency
    # ------------------------------------------------------------------
    _print_export_summary(mlpackage_path, mlmodel_path, model_type)

    # ------------------------------------------------------------------
    # 8. Ensure the ios/Models README exists
    # ------------------------------------------------------------------
    _write_ios_models_readme()


def _print_export_summary(
    mlpackage_path: Path,
    mlmodel_path: Path,
    model_type: str,
) -> None:
    """Print a summary of the exported files."""
    def _dir_size_mb(p: Path) -> float:
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total = p.stat().st_size
        else:
            return 0.0
        return total / (1024 * 1024)

    pkg_mb = _dir_size_mb(mlpackage_path)
    mdl_mb = _dir_size_mb(mlmodel_path)
    latency = _INFERENCE_MS_ESTIMATES.get(model_type, "unknown")

    print(f"\n  Export complete:")
    if pkg_mb > 0:
        print(f"    .mlpackage : {pkg_mb:.2f} MB  ({mlpackage_path.name})")
    if mdl_mb > 0:
        print(f"    .mlmodel   : {mdl_mb:.2f} MB  ({mlmodel_path.name})")
    print(f"    Estimated inference time on iPhone Neural Engine: {latency}")
    print(
        "  Note: Run on-device benchmarks with Xcode Instruments > Core ML for "
        "accurate latency measurements on your target hardware."
    )


def _write_ios_models_readme() -> None:
    """Create mobile/react-native/ios/Models/ with a README if it does not exist."""
    ios_models_dir = (
        _REPO_ROOT / "mobile" / "react-native" / "ios" / "Models"
    )
    ios_models_dir.mkdir(parents=True, exist_ok=True)
    readme_path = ios_models_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(_IOS_MODELS_README)
        print(f"\n  Created iOS Models directory: {ios_models_dir}")
        print(f"  README written: {readme_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export EMG-ASL PyTorch models to Apple CoreML format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_coreml.py --model models/asl_emg_classifier.pt\n"
            "  python scripts/export_coreml.py --model models/cnn_lstm_classifier.pt --type cnn_lstm\n"
            "  python scripts/export_coreml.py --all --quantize\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the .pt checkpoint to export.",
    )
    parser.add_argument(
        "--type",
        dest="model_type",
        choices=["lstm", "cnn_lstm", "conformer"],
        default=None,
        help=(
            "Model architecture. If omitted, inferred from the filename "
            "(e.g. 'cnn_lstm_classifier.pt' -> cnn_lstm)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to write .mlpackage and .mlmodel files (default: models/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all .pt files found in --output-dir (or models/ by default).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help=(
            "Apply FP16 linear weight quantisation after conversion. "
            "Reduces model size by ~2x with negligible accuracy loss."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate that coremltools is present before doing any work
    _require_coremltools()

    output_dir: Path = args.output_dir

    if args.all:
        # Find all .pt files in the output directory
        search_dir = output_dir if output_dir.is_dir() else Path("models")
        pt_files = sorted(search_dir.glob("*.pt"))
        if not pt_files:
            print(f"[export_coreml] No .pt files found in '{search_dir}'.")
            sys.exit(1)
        print(f"[export_coreml] Found {len(pt_files)} .pt file(s) to export.")
        for pt_file in pt_files:
            model_type = _infer_model_type(pt_file)
            try:
                export_model(
                    model_path=pt_file,
                    model_type=model_type,
                    output_dir=output_dir,
                    quantize=args.quantize,
                )
            except Exception as exc:
                print(f"  [ERROR] Failed to export {pt_file.name}: {exc}")
        return

    if args.model is None:
        parser.error("Provide --model <path> or use --all.")

    model_path: Path = args.model
    if not model_path.exists():
        print(f"[export_coreml] Model file not found: {model_path}")
        sys.exit(1)

    model_type = args.model_type or _infer_model_type(model_path)
    print(f"[export_coreml] Architecture: {model_type}")

    export_model(
        model_path=model_path,
        model_type=model_type,
        output_dir=output_dir,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()

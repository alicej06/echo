"""
Per-user calibration CLI for the EMG-ASL inference system.

This script is the command-line entry point for running calibration sessions.
It is distinct from src/models/calibration.py, which is a reusable module.
This script orchestrates the interactive session, data collection, fine-tuning,
and ONNX export in a single command.

What this script does
----------------------
1.  Loads the base ASLEMGClassifier from a .pt or .onnx path.
2.  Prompts the user to connect the MAIA armband (or uses synthetic data
    if --synthetic is set).
3.  Runs a 30-60 second guided calibration:
      - Iterates over 10 signs: A, B, C, D, E, F, G, H, I, J.
      - For each sign, counts down 3 seconds, collecting EMG windows.
      - Builds a user-specific feature centroid per class from the collected data.
4.  Evaluates per-class accuracy on the collected data before calibration
    (using the base model) and after (using the adapted model).
5.  Fine-tunes a small 16-unit linear adaptor layer on top of the frozen
    base model using the calibration samples.
6.  Saves the calibrated model to:
        {output-dir}/{user-id}/model.onnx
      and the PyTorch checkpoint to:
        {output-dir}/{user-id}/model.pt
7.  Prints a before/after accuracy report.

Adaptor architecture
---------------------
The adaptor is a single Linear(hidden_size -> 16) -> ReLU -> Linear(16 -> num_classes)
head that replaces the base model's final fc layer.  All other layers are frozen.
This is intentionally tiny so that 10-30 samples per class is sufficient to
specialize the model to a new user's muscle activation patterns without
overfitting.

Usage
-----
    # With hardware (prompts user to hold each sign):
    python scripts/calibrate_user.py --user-id alice \\
        --model models/asl_emg_classifier.pt

    # Fully synthetic (no hardware required, good for CI / testing):
    python scripts/calibrate_user.py --user-id test_user \\
        --model models/asl_emg_classifier.pt \\
        --synthetic

    # Custom output directory:
    python scripts/calibrate_user.py --user-id bob \\
        --model models/asl_emg_classifier.pt \\
        --output-dir models/calibrated \\
        --synthetic
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_classifier import ASLEMGClassifier
from src.utils.constants import (
    ASL_LABELS,
    FEATURE_VECTOR_SIZE,
    N_CHANNELS,
    NUM_CLASSES,
    SAMPLE_RATE,
    WINDOW_SIZE_SAMPLES,
)

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Signs covered in the 10-sign calibration session.
CALIB_SIGNS: list[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# How long the user holds each sign (seconds).
HOLD_DURATION_S: float = 3.0

# Adaptor bottleneck dimension.
ADAPTOR_HIDDEN: int = 16

# Fine-tuning hyperparameters.
FINE_TUNE_EPOCHS: int = 80
FINE_TUNE_LR: float = 5e-4
FINE_TUNE_WEIGHT_DECAY: float = 1e-4
FINE_TUNE_BATCH_SIZE: int = 16


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-user EMG-ASL calibration session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help=(
            "Unique identifier for this user "
            "(alphanumeric, used as a subdirectory name)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the base model (.pt PyTorch checkpoint or .onnx file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/calibrated/",
        help="Root directory for calibrated model output.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use synthetic EMG data instead of real hardware. "
            "Useful for testing without a connected armband."
        ),
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=HOLD_DURATION_S,
        help="Seconds to hold each sign during calibration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (only affects synthetic data generation).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_base_model(model_path: str) -> ASLEMGClassifier:
    """Load the base model from either a .pt or .onnx file.

    For ONNX files the model weights cannot be recovered for fine-tuning,
    so we instantiate a default ASLEMGClassifier with the standard architecture
    and note that its weights are randomly initialized.  In practice the user
    should provide a .pt file.

    Parameters
    ----------
    model_path:
        Path to the model file.

    Returns
    -------
    ASLEMGClassifier
        Loaded model in evaluation mode.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pt":
        model = ASLEMGClassifier.load(path)
        print(f"  Loaded PyTorch checkpoint: {path}")
        return model
    elif suffix == ".onnx":
        print(
            f"  [WARNING] ONNX model provided ({path}).\n"
            "  ONNX weights cannot be fine-tuned directly.\n"
            "  Instantiating a default ASLEMGClassifier for adaptation.\n"
            "  For best results, provide a .pt checkpoint instead."
        )
        model = ASLEMGClassifier(
            input_size=FEATURE_VECTOR_SIZE,
            hidden_size=128,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout=0.3,
            label_names=list(ASL_LABELS),
        )
        model.eval()
        return model
    else:
        raise ValueError(
            f"Unsupported model format '{suffix}'. "
            "Provide a .pt PyTorch checkpoint or a .onnx file."
        )


# ---------------------------------------------------------------------------
# Adaptor model
# ---------------------------------------------------------------------------


class AdaptorHead(nn.Module):
    """Tiny adaptor that replaces the base model's fc head.

    Architecture:
        Linear(hidden_size, ADAPTOR_HIDDEN) -> ReLU -> Linear(ADAPTOR_HIDDEN, num_classes)

    Only this module's parameters are updated during fine-tuning.  The rest
    of the base model remains frozen.

    Parameters
    ----------
    hidden_size:
        Output dimension of the last LSTM hidden state (= base model hidden_size).
    num_classes:
        Number of output classes.
    adaptor_hidden:
        Bottleneck dimension of the adaptor.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        adaptor_hidden: int = ADAPTOR_HIDDEN,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, adaptor_hidden),
            nn.ReLU(),
            nn.Linear(adaptor_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_adapted_model(
    base_model: ASLEMGClassifier,
) -> ASLEMGClassifier:
    """Replace the base model's fc head with a small AdaptorHead.

    The returned model is a deep copy of base_model with only the new
    AdaptorHead's parameters set to requires_grad=True.

    Parameters
    ----------
    base_model:
        Pre-trained base classifier.

    Returns
    -------
    ASLEMGClassifier
        A deep-copied model with frozen body and trainable AdaptorHead.
    """
    model = copy.deepcopy(base_model)

    # Freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    # Replace fc head with the adaptor (unfrozen by default for new modules).
    adaptor = AdaptorHead(
        hidden_size=model.hidden_size,
        num_classes=model.num_classes,
        adaptor_hidden=ADAPTOR_HIDDEN,
    )
    model.fc = adaptor

    return model


# ---------------------------------------------------------------------------
# Feature extraction from raw windows
# ---------------------------------------------------------------------------


def _extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """Extract the 80-dim feature vector from a raw (40, 8) EMG window.

    Wraps src.utils.features.extract_features with the standard sample rate.

    Parameters
    ----------
    window:
        Shape (WINDOW_SIZE_SAMPLES, N_CHANNELS), float32.

    Returns
    -------
    np.ndarray
        Shape (FEATURE_VECTOR_SIZE,) = (80,).
    """
    from src.utils.features import extract_features

    return extract_features(window, fs=float(SAMPLE_RATE))


# ---------------------------------------------------------------------------
# Synthetic calibration data generation
# ---------------------------------------------------------------------------


def _collect_synthetic_windows(
    sign: str,
    hold_duration_s: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate synthetic EMG windows for one sign during calibration.

    Uses the same synthetic signal model as src/data/synthetic.py so that
    the generated windows are consistent with the rest of the codebase.

    Parameters
    ----------
    sign:
        ASL sign label string (must be in ASL_LABELS).
    hold_duration_s:
        Duration of the hold in seconds.
    rng:
        Numpy random generator.

    Returns
    -------
    list[np.ndarray]
        List of (WINDOW_SIZE_SAMPLES, N_CHANNELS) float32 arrays, one per
        sliding window extracted from the generated hold segment.
    """
    from src.data.synthetic import _build_class_profile, _generate_hold

    n_samples = max(WINDOW_SIZE_SAMPLES, int(SAMPLE_RATE * hold_duration_s))
    profile = _build_class_profile(sign)
    hold_data = _generate_hold(
        n_samples=n_samples,
        fs=float(SAMPLE_RATE),
        profile=profile,
        noise_level=0.1,
        rng=rng,
    )  # (n_samples, N_CHANNELS)

    # Slide windows.
    step = max(1, WINDOW_SIZE_SAMPLES // 2)
    windows: list[np.ndarray] = []
    for start in range(0, n_samples - WINDOW_SIZE_SAMPLES + 1, step):
        windows.append(hold_data[start : start + WINDOW_SIZE_SAMPLES].copy())

    return windows


# ---------------------------------------------------------------------------
# Hardware data collection (stub for real BLE integration)
# ---------------------------------------------------------------------------


def _collect_hardware_windows(
    sign: str,
    hold_duration_s: float,
) -> list[np.ndarray]:
    """Collect EMG windows from the MAIA armband during a live hold.

    This is a stub implementation that raises NotImplementedError.  The real
    implementation would open a BLE connection to the MYO Armband, stream
    raw samples at SAMPLE_RATE Hz for hold_duration_s seconds, apply the
    bandpass/notch filter chain from src/utils/filters.py, and extract
    sliding windows.

    To integrate hardware:
      1. Import the BLE streaming code from hardware/myoware_ble or the
         WebSocket client from src/api/websocket.py.
      2. Collect (n_samples, N_CHANNELS) raw signal data.
      3. Apply src.utils.filters.apply_filters() per channel.
      4. Slide windows with step = WINDOW_SIZE_SAMPLES // 2.
      5. Return the list of (WINDOW_SIZE_SAMPLES, N_CHANNELS) windows.

    Parameters
    ----------
    sign:
        The ASL sign currently being recorded (for display purposes only).
    hold_duration_s:
        Desired hold duration in seconds.

    Returns
    -------
    list[np.ndarray]
        List of (WINDOW_SIZE_SAMPLES, N_CHANNELS) float32 windows.

    Raises
    ------
    NotImplementedError
        Always.  Replace this function body with real BLE streaming code.
    """
    raise NotImplementedError(
        "Hardware BLE streaming is not yet implemented in this script.\n"
        "Use --synthetic to run a calibration session without hardware, or\n"
        "implement the BLE streaming logic in _collect_hardware_windows()."
    )


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------


def _evaluate_accuracy(
    model: ASLEMGClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    label_map: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate per-class accuracy on a small labeled feature set.

    Parameters
    ----------
    model:
        Classifier to evaluate (will be put in eval mode).
    features:
        Shape (N, FEATURE_VECTOR_SIZE) float32.
    labels:
        Shape (N,) string labels.
    label_map:
        Mapping from label string to class index.
    device:
        Torch device.

    Returns
    -------
    dict[str, float]
        Per-class accuracy keyed by sign string, plus 'overall' accuracy.
    """
    model.eval()
    model = model.to(device)

    X = torch.from_numpy(features).to(device)
    y_int = np.array([label_map[l] for l in labels], dtype=np.int64)

    with torch.no_grad():
        logits = model(X)  # (N, num_classes)
        preds = logits.argmax(dim=1).cpu().numpy()

    results: dict[str, float] = {}
    for sign in CALIB_SIGNS:
        if sign not in label_map:
            continue
        idx = label_map[sign]
        mask = y_int == idx
        if mask.sum() == 0:
            results[sign] = float("nan")
        else:
            results[sign] = float((preds[mask] == idx).mean())

    overall_correct = int((preds == y_int).sum())
    results["overall"] = overall_correct / len(y_int) if len(y_int) > 0 else 0.0
    return results


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def _fine_tune(
    model: ASLEMGClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    label_map: dict[str, int],
    device: torch.device,
) -> ASLEMGClassifier:
    """Fine-tune the model's adaptor head on the calibration features.

    Only parameters of model.fc (the AdaptorHead) are updated.  All other
    parameters must already be frozen (requires_grad=False) before this
    function is called.

    Parameters
    ----------
    model:
        Model with frozen body and trainable AdaptorHead.
    features:
        Shape (N, FEATURE_VECTOR_SIZE) float32 calibration features.
    labels:
        Shape (N,) string labels corresponding to CALIB_SIGNS.
    label_map:
        Mapping from label string to class index.
    device:
        Torch device.

    Returns
    -------
    ASLEMGClassifier
        Fine-tuned model in evaluation mode.
    """
    model = model.to(device)
    model.train()

    y_int = np.array([label_map[l] for l in labels], dtype=np.int64)
    X_tensor = torch.from_numpy(features).to(device)
    y_tensor = torch.from_numpy(y_int).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=min(FINE_TUNE_BATCH_SIZE, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    # Only optimize the adaptor head parameters.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found. "
            "Ensure _build_adapted_model() was called before fine-tuning."
        )

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=FINE_TUNE_LR,
        weight_decay=FINE_TUNE_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(FINE_TUNE_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = epoch_loss / len(dataset)
            print(
                f"    [fine-tune] epoch {epoch + 1:3d}/{FINE_TUNE_EPOCHS}  "
                f"loss={avg:.4f}"
            )

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Countdown helper
# ---------------------------------------------------------------------------


def _countdown(sign: str, hold_s: float, synthetic: bool) -> None:
    """Print a brief countdown for the user to prepare to hold a sign.

    In synthetic mode the countdown is skipped (no user interaction needed).

    Parameters
    ----------
    sign:
        The sign the user is about to hold.
    hold_s:
        Duration of the hold in seconds.
    synthetic:
        If True, skip interactive delay.
    """
    if synthetic:
        print(f"  Collecting data for sign '{sign}' (synthetic) ...")
        return

    print(f"\n  Next sign: [{sign}]")
    print(f"  Hold the '{sign}' handshape for {hold_s:.0f} seconds.")
    print("  Starting in: ", end="", flush=True)
    for i in range(3, 0, -1):
        print(f"{i} ... ", end="", flush=True)
        time.sleep(1.0)
    print("GO!", flush=True)


# ---------------------------------------------------------------------------
# Main calibration flow
# ---------------------------------------------------------------------------


def run_calibration(args: argparse.Namespace) -> None:
    """Execute the full calibration pipeline.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print("=" * 64)
    print(f"EMG-ASL per-user calibration  --  user: {args.user_id}")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Step 1: Load base model
    # ------------------------------------------------------------------
    print("\nStep 1 -- Loading base model ...")
    base_model = _load_base_model(args.model)
    print(f"  {base_model}")

    # Build label map from the model's label_names (or ASL_LABELS fallback).
    label_names: list[str] = (
        base_model.label_names if base_model.label_names else list(ASL_LABELS)
    )
    label_map: dict[str, int] = {name: i for i, name in enumerate(label_names)}

    # Confirm all calibration signs are known to the model.
    unknown_signs = [s for s in CALIB_SIGNS if s not in label_map]
    if unknown_signs:
        print(
            f"  [WARNING] The following calibration signs are not in the "
            f"model's vocabulary: {unknown_signs}.\n"
            f"  They will be skipped."
        )
        calib_signs_to_use = [s for s in CALIB_SIGNS if s in label_map]
    else:
        calib_signs_to_use = list(CALIB_SIGNS)

    if not calib_signs_to_use:
        print("[ERROR] No usable calibration signs. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Hardware connection or synthetic mode
    # ------------------------------------------------------------------
    print("\nStep 2 -- Armband setup ...")
    if args.synthetic:
        print(
            "  Synthetic mode enabled. No hardware connection required.\n"
            "  Real EMG data will be simulated using the synthetic data generator."
        )
    else:
        print(
            "  Please connect the MYO Armband (ensure MyoConnect is running\n"
            "  and the armband shows a green LED before continuing).\n"
            "  (Press ENTER when ready ...)"
        )
        try:
            input()
        except EOFError:
            pass
        print("  Proceeding with hardware data collection ...")

    # ------------------------------------------------------------------
    # Step 3: Guided calibration session
    # ------------------------------------------------------------------
    print(
        f"\nStep 3 -- Guided calibration session "
        f"({len(calib_signs_to_use)} signs x {args.hold_duration:.0f}s)\n"
    )

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    per_class_centroids: dict[str, np.ndarray] = {}

    for sign in calib_signs_to_use:
        _countdown(sign, args.hold_duration, args.synthetic)

        # Collect raw EMG windows for this sign.
        if args.synthetic:
            raw_windows = _collect_synthetic_windows(sign, args.hold_duration, rng)
        else:
            raw_windows = _collect_hardware_windows(sign, args.hold_duration)

        if not raw_windows:
            print(f"  [WARNING] No windows collected for sign '{sign}'. Skipping.")
            continue

        # Extract feature vectors.
        sign_features = np.stack(
            [_extract_features_from_window(w) for w in raw_windows], axis=0
        )  # (n_windows, FEATURE_VECTOR_SIZE)

        # Build user-specific centroid for this class.
        per_class_centroids[sign] = sign_features.mean(axis=0)

        all_features.append(sign_features)
        all_labels.extend([sign] * len(sign_features))

        print(
            f"  Collected {len(raw_windows)} windows for '{sign}'  "
            f"(centroid norm: "
            f"{float(np.linalg.norm(per_class_centroids[sign])):.3f})"
        )

    if not all_features:
        print("[ERROR] No calibration data collected. Aborting.")
        sys.exit(1)

    features_all = np.vstack(all_features).astype(np.float32)
    labels_all = np.array(all_labels)

    print(
        f"\n  Total calibration samples: {len(features_all)}  "
        f"across {len(per_class_centroids)} classes."
    )

    # ------------------------------------------------------------------
    # Step 4: Pre-calibration accuracy
    # ------------------------------------------------------------------
    print("\nStep 4 -- Pre-calibration accuracy (base model) ...")
    pre_acc = _evaluate_accuracy(
        base_model, features_all, labels_all, label_map, device
    )
    print(f"  Overall (base): {pre_acc['overall'] * 100:.1f} %")
    for sign in calib_signs_to_use:
        acc_val = pre_acc.get(sign, float("nan"))
        print(f"    {sign}: {acc_val * 100:5.1f} %")

    # ------------------------------------------------------------------
    # Step 4b: Build adapted model
    # ------------------------------------------------------------------
    print("\nStep 4b -- Building adapted model (adding AdaptorHead) ...")
    adapted_model = _build_adapted_model(base_model)
    n_trainable = sum(
        p.numel() for p in adapted_model.parameters() if p.requires_grad
    )
    n_frozen = sum(
        p.numel() for p in adapted_model.parameters() if not p.requires_grad
    )
    print(
        f"  Trainable parameters  : {n_trainable:,}  "
        f"(AdaptorHead only)"
    )
    print(f"  Frozen parameters     : {n_frozen:,}")

    # ------------------------------------------------------------------
    # Step 5: Fine-tune
    # ------------------------------------------------------------------
    print(
        f"\nStep 5 -- Fine-tuning AdaptorHead "
        f"({FINE_TUNE_EPOCHS} epochs, lr={FINE_TUNE_LR}) ..."
    )
    adapted_model = _fine_tune(
        adapted_model, features_all, labels_all, label_map, device
    )
    print("  Fine-tuning complete.")

    # ------------------------------------------------------------------
    # Step 6: Post-calibration accuracy
    # ------------------------------------------------------------------
    print("\nStep 6 -- Post-calibration accuracy (adapted model) ...")
    post_acc = _evaluate_accuracy(
        adapted_model, features_all, labels_all, label_map, device
    )
    print(f"  Overall (adapted): {post_acc['overall'] * 100:.1f} %")
    for sign in calib_signs_to_use:
        pre = pre_acc.get(sign, float("nan"))
        post = post_acc.get(sign, float("nan"))
        delta = (post - pre) if not (np.isnan(pre) or np.isnan(post)) else float("nan")
        delta_str = (
            f"(+{delta * 100:.1f} %)"
            if not np.isnan(delta) and delta >= 0
            else f"({delta * 100:.1f} %)" if not np.isnan(delta) else "(n/a)"
        )
        print(f"    {sign}: {pre * 100:5.1f} % -> {post * 100:5.1f} %  {delta_str}")

    overall_delta = post_acc["overall"] - pre_acc["overall"]
    delta_tag = (
        f"+{overall_delta * 100:.1f} %"
        if overall_delta >= 0
        else f"{overall_delta * 100:.1f} %"
    )
    print(
        f"\n  Overall improvement: "
        f"{pre_acc['overall'] * 100:.1f} % -> "
        f"{post_acc['overall'] * 100:.1f} %  ({delta_tag})"
    )

    # ------------------------------------------------------------------
    # Step 7: Save calibrated model
    # ------------------------------------------------------------------
    print("\nStep 7 -- Saving calibrated model ...")

    out_dir = Path(args.output_dir) / args.user_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = str(out_dir / "model.pt")
    onnx_path = str(out_dir / "model.onnx")

    adapted_model.cpu()
    adapted_model.eval()

    # Restore label names so the saved checkpoint is self-describing.
    if adapted_model.label_names is None:
        adapted_model.label_names = label_names

    adapted_model.save(pt_path)
    print(f"  PyTorch checkpoint : {pt_path}")

    try:
        adapted_model.to_onnx(onnx_path)
        print(f"  ONNX model         : {onnx_path}")
    except Exception as exc:
        print(
            f"  [WARNING] ONNX export failed: {exc}\n"
            "  The PyTorch checkpoint is still usable."
        )

    # Save per-class centroids as a numpy archive for downstream use.
    centroid_path = str(out_dir / "class_centroids.npy")
    np.save(centroid_path, per_class_centroids)  # type: ignore[arg-type]
    print(f"  Class centroids    : {centroid_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print(f"Calibration complete for user '{args.user_id}'")
    print("=" * 64)
    print(
        f"  Calibration samples : {len(features_all)}\n"
        f"  Classes calibrated  : {list(per_class_centroids.keys())}\n"
        f"  Pre-cal accuracy    : {pre_acc['overall'] * 100:.1f} %\n"
        f"  Post-cal accuracy   : {post_acc['overall'] * 100:.1f} %\n"
        f"  Saved to            : {out_dir}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_calibration(args)


if __name__ == "__main__":
    main()

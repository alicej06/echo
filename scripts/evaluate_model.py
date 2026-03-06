"""
Evaluate a saved EMG-ASL model on a data directory.

Supports both PyTorch (.pt) and ONNX (.onnx) model files. Auto-detects
format from file extension.

Usage (from project root)
--------------------------
    python scripts/evaluate_model.py --model models/asl_emg_classifier.pt --data-dir data/raw/
    python scripts/evaluate_model.py --model models/asl_emg_classifier.onnx --data-dir data/raw/
    python scripts/evaluate_model.py --model models/asl_emg_classifier.pt \\
        --data-dir data/raw/ --save-plot results/confusion_matrix.png

Printed output
--------------
  - Overall accuracy
  - Per-class precision, recall, F1 (sklearn classification report)
  - Confusion matrix in text format
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.*`` imports work when
# invoked as ``python scripts/evaluate_model.py`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.loader import create_windows, load_dataset
from src.utils.constants import ASL_LABELS


# ---------------------------------------------------------------------------
# PyTorch inference
# ---------------------------------------------------------------------------


def _evaluate_pt(
    model_path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Load a .pt checkpoint and evaluate using src.models.train.evaluate_model."""
    from src.models.lstm_classifier import ASLEMGClassifier
    from src.models.train import evaluate_model

    print(f"Loading PyTorch model from {model_path} …")
    model = ASLEMGClassifier.load(model_path)
    model.eval()
    print(f"  Model architecture : {model}")
    print(f"  Input size         : {model.input_size}")
    print(f"  Num classes        : {model.num_classes}")

    metrics = evaluate_model(
        model=model,
        test_windows=windows,
        test_labels=labels,
        label_names=list(ASL_LABELS),
    )
    return metrics


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------


def _evaluate_onnx(
    model_path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Load an .onnx model via onnxruntime and evaluate."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: pip install onnxruntime"
        )

    print(f"Loading ONNX model from {model_path} …")
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress verbose logs
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"  ONNX input  : '{input_name}'  shape={session.get_inputs()[0].shape}")
    print(f"  ONNX output : '{output_name}' shape={session.get_outputs()[0].shape}")

    # Flatten 3-D windows -> 2-D to match ONNX export (model sees flat vectors)
    if windows.ndim == 3:
        N, T, C = windows.shape
        X_flat = windows.reshape(N, T * C).astype(np.float32)
    else:
        X_flat = windows.astype(np.float32)

    # Run inference in batches of 256
    all_preds: List[int] = []
    batch_size = 256
    for start in range(0, len(X_flat), batch_size):
        batch = X_flat[start: start + batch_size]
        logits = session.run([output_name], {input_name: batch})[0]  # (B, C)
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())

    y_pred_int = np.array(all_preds, dtype=np.int64)

    # Encode ground-truth labels
    label_map = {name: i for i, name in enumerate(ASL_LABELS)}
    try:
        y_true_int = np.array([label_map[l] for l in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown label in test set: {exc}") from exc

    present_classes = sorted(set(y_true_int.tolist()))
    present_names = [ASL_LABELS[i] for i in present_classes if i < len(ASL_LABELS)]

    acc = accuracy_score(y_true_int, y_pred_int)
    report_str = classification_report(
        y_true_int,
        y_pred_int,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_int, y_pred_int, labels=present_classes)

    from sklearn.metrics import precision_recall_fscore_support

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_int,
        y_pred_int,
        labels=present_classes,
        zero_division=0,
    )
    per_class = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for name, p, r, f in zip(present_names, prec, rec, f1)
    }

    print(f"[evaluate_model] Accuracy: {acc:.4f}")
    print(report_str)

    return {
        "accuracy": acc,
        "report": report_str,
        "per_class": per_class,
        "confusion_matrix": cm,
        "y_true": y_true_int,
        "y_pred": y_pred_int,
        "present_names": present_names,
    }


# ---------------------------------------------------------------------------
# Confusion matrix text renderer
# ---------------------------------------------------------------------------


def _print_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """Print the confusion matrix in a compact text table."""
    n = len(class_names)
    # Abbreviate long names (e.g. THANK_YOU -> THN)
    abbrevs = [name[:4] for name in class_names]
    col_w = max(len(a) for a in abbrevs) + 1

    # Header
    header = " " * (col_w + 1) + "".join(f"{a:>{col_w}}" for a in abbrevs)
    print(header)
    print("-" * len(header))

    for i, row_name in enumerate(abbrevs):
        row_str = f"{row_name:>{col_w}} |"
        for j in range(n):
            cell = cm[i, j]
            row_str += f"{cell:>{col_w}}"
        print(row_str)


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Save a colour-coded confusion matrix image to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping confusion matrix plot.")
        return

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    fig_size = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    # Annotate cells when there are few classes
    if n <= 20:
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6,
                )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] Confusion matrix plot -> {out.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved EMG-ASL model on a data directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_model.py --model models/asl_emg_classifier.pt --data-dir data/raw/
  python scripts/evaluate_model.py --model models/asl_emg_classifier.onnx --data-dir data/raw/
  python scripts/evaluate_model.py --model models/asl_emg_classifier.pt \\
      --data-dir data/raw/ --save-plot results/confusion_matrix.png
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a .pt (PyTorch) or .onnx model file.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        dest="data_dir",
        help="Directory containing session CSV files to evaluate on.",
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        dest="save_plot",
        metavar="PATH",
        help="Optional: save confusion matrix plot to this path (e.g. results/cm.png).",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/3  Loading dataset")
    print("=" * 60)
    df = load_dataset(args.data_dir)

    print()
    print("=" * 60)
    print("Step 2/3  Creating sliding windows")
    print("=" * 60)
    windows, labels = create_windows(df)
    print(f"  Windows : {windows.shape}  ({len(np.unique(labels))} unique labels)")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3/3  Running inference")
    print("=" * 60)

    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        metrics = _evaluate_pt(model_path, windows, labels)
    elif suffix == ".onnx":
        metrics = _evaluate_onnx(model_path, windows, labels)
    else:
        print(
            f"[ERROR] Unsupported model format '{suffix}'. "
            "Use a .pt or .onnx file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Confusion matrix text output
    # ------------------------------------------------------------------
    present_names = metrics.get(
        "present_names",
        [ASL_LABELS[i] for i in sorted(set(metrics["y_true"].tolist())) if i < len(ASL_LABELS)],
    )
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 60)
    _print_confusion_matrix(cm, present_names)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Overall accuracy : {metrics['accuracy']:.4f}  ({metrics['accuracy'] * 100:.2f} %)")
    print()
    print("Per-class F1 scores:")
    for cls_name, vals in sorted(metrics["per_class"].items()):
        print(f"  {cls_name:>12s} : F1={vals['f1']:.3f}  "
              f"P={vals['precision']:.3f}  R={vals['recall']:.3f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Optional: save confusion matrix plot
    # ------------------------------------------------------------------
    if args.save_plot:
        _save_confusion_matrix_plot(cm, present_names, args.save_plot)


if __name__ == "__main__":
    main()

"""
evaluate.py
-----------
Per-user LOO evaluation of DyFAV matching the paper's exact protocol,
plus confusion matrix output.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --top-k 327
    python scripts/evaluate.py --top-k 64 150 240 327 510   # sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.preprocess import load_dataset
from scripts.train_dyfav import train_dyfav, predict_dyfav, ALL_LETTERS, TOP_K

DATA_DIR   = ROOT / "data" / "raw" / "dyfav"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = [l.upper() for l in ALL_LETTERS]


def evaluate_top_k(top_k: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Run per-user LOO-CV at a given top_k. Returns (accuracy, y_true, y_pred)."""
    user_data = load_dataset(str(DATA_DIR), per_user=True)
    y_true_all, y_pred_all = [], []

    for _uid, (X, y, rec_ids) in sorted(user_data.items()):
        recs_per_letter: dict[int, list] = {}
        for rec_id in np.unique(rec_ids):
            label = y[rec_ids == rec_id][0]
            recs_per_letter.setdefault(int(label), []).append(rec_id)

        n_folds = max(len(v) for v in recs_per_letter.values())

        for held_pos in range(n_folds):
            test_recs  = [recs[held_pos]
                          for recs in recs_per_letter.values()
                          if held_pos < len(recs)]
            test_mask  = np.isin(rec_ids, test_recs)
            train_mask = ~test_mask
            if not (train_mask.any() and test_mask.any()):
                continue

            model = train_dyfav(X[train_mask], y[train_mask], top_k=top_k)
            for i in np.where(test_mask)[0]:
                pred, _ = predict_dyfav(model, X[i])
                y_true_all.append(int(y[i]))
                y_pred_all.append(pred)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    acc    = (y_true == y_pred).mean()
    return acc, y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, nargs="+", default=[TOP_K],
                        help="Top-K feature counts to evaluate (default: 327)")
    return parser.parse_args()


def main():
    args = parse_args()
    top_ks = args.top_k

    print(f"\n{'='*55}")
    print(f"  ECHO — DyFAV Evaluation (per-user LOO-CV)")
    print(f"{'='*55}")

    results = {}
    for k in top_ks:
        print(f"\nEvaluating top_k={k}...")
        acc, y_true, y_pred = evaluate_top_k(k)
        results[k] = (acc, y_true, y_pred)
        print(f"  top_k={k:4d}  accuracy={acc:.1%}")

    # Detailed report for the best/primary k
    primary_k   = top_ks[0]
    acc, y_true, y_pred = results[primary_k]

    print(f"\n{'='*55}")
    print(f"  Overall accuracy (top_k={primary_k}): {acc:.1%}")
    print(f"  Paper reports: 95.36%")
    print(f"{'='*55}\n")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=2))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
        ax=ax, linewidths=0.3, annot_kws={"size": 7},
    )
    ax.set_title(f"DyFAV Confusion Matrix — {acc:.1%} (top_k={primary_k})", fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[saved] Confusion matrix → {cm_path}")

    # Accuracy vs features plot (if multiple k values)
    if len(top_ks) > 1:
        ks   = sorted(results.keys())
        accs = [results[k][0] for k in ks]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, [a * 100 for a in accs], "o-", color="#06b6d4", linewidth=2)
        ax.axhline(95.36, color="orange", linestyle="--", label="Paper (95.36%)")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("DyFAV: Accuracy vs Number of Features")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "accuracy_vs_features.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[saved] Accuracy plot → {plot_path}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

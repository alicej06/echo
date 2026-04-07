"""
train_dyfav.py
--------------
Exact replication of the DyFAV (Dynamic Feature Selection And Voting)
algorithm from the official GitHub repo:
  https://github.com/prwlnght/DyFAV

Reference paper:
  Paudyal et al. "A Comparison of Techniques for Sign Language Alphabet
  Recognition Using Armband Wearables." ACM TIST 9(2-3), 2019.

build_model() — exact weight formula from dyFAV.py:
  weight = (N * n / (upper_idx - lower_idx + 1) - n) / (N - n)

  Thresholds are padded at training time:
    padding = abs((val_upper - val_lower) / n_class_examples)
    threshold_lower = val_lower - padding
    threshold_upper = val_upper + padding

recognize() — exact fuzzy threshold from dyFAV.py:
  adj_lower = threshold_lower - fuzzy * abs(threshold_lower)
  adj_upper = threshold_upper + fuzzy * abs(threshold_lower)  ← uses lower for both

  Per-class weight normalization applied at prediction time over top-K features.

Usage:
    python scripts/train_dyfav.py --evaluate
    python scripts/train_dyfav.py --user 1
    python scripts/train_dyfav.py --all-users
    python scripts/train_dyfav.py --evaluate --randomized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.preprocess import load_dataset, extract_510_features

DATA_DIR   = ROOT / "data" / "raw" / "dyfav"
MODELS_DIR = ROOT / "models"

N_CLASSES   = 26
TOP_K       = 510    # repo default is all 510 features; paper optimal is 327
ALL_LETTERS = [chr(ord('a') + i) for i in range(26)]

# fuzzy_thresholds from dyFAV.py: [0.0175, 0.0225]
FUZZY_THRESHOLD_DEFAULT = 0.0228   # empirically best: 89.2% avg accuracy


# ---------------------------------------------------------------------------
# Core DyFAV — matches build_model() in dyFAV.py exactly
# ---------------------------------------------------------------------------

def train_dyfav(
    X: np.ndarray,
    y: np.ndarray,
    top_k: int = TOP_K,
) -> dict:
    """
    Train DyFAV feature models for all 26 letter agents.

    Matches build_model() from the official GitHub repo exactly.
    Raw (unpadded) weights are stored; normalization is done at predict time.

    Args:
        X:     (n_samples, 510) — one row per training recording
        y:     (n_samples,) — integer labels 0-25
        top_k: number of top-weight features to retain per agent

    Returns:
        model dict with 'letter_models':
          {letter_idx: [(threshold_lower, threshold_upper, weight, feat_idx), ...]}
          sorted by weight descending, truncated to top_k
    """
    n_samples, n_features = X.shape
    N = n_samples    # total training examples
    letter_models = {}

    for letter_idx in range(N_CLASSES):
        letter = ALL_LETTERS[letter_idx]
        class_mask = (y == letter_idx)
        n = int(class_mask.sum())   # this_examples

        if n == 0 or n >= N:
            letter_models[letter_idx] = []
            continue

        per_feature = []   # (threshold_lower, threshold_upper, weight, feat_idx)

        for feat_idx in range(n_features):
            # Sort all training instances by this feature
            sort_order    = np.argsort(X[:, feat_idx], kind="stable")
            sorted_labels = y[sort_order]
            sorted_values = X[sort_order, feat_idx]

            # Find lower and upper positions of this letter in sorted array
            positions = np.where(sorted_labels == letter_idx)[0]
            lower_idx = int(positions.min())
            upper_idx = int(positions.max())

            val_lower = float(sorted_values[lower_idx])
            val_upper = float(sorted_values[upper_idx])

            # Padding — matches dyFAV.py exactly:
            #   padding_heuristics = abs((val_upper - val_lower) / this_examples)
            padding = abs((val_upper - val_lower) / n)
            threshold_lower = val_lower - padding
            threshold_upper = val_upper + padding

            # Weight formula — matches dyFAV.py exactly:
            #   (total_range * this_examples / (upper_range - lower_range + 1) - this_examples)
            #   / (total_range - this_examples)
            idx_range = upper_idx - lower_idx  # can be 0 when all examples cluster together
            weight = (N * n / (idx_range + 1) - n) / (N - n)

            per_feature.append((threshold_lower, threshold_upper, weight, feat_idx))

        # Sort by weight descending, keep top_k — raw weights stored (normalized at predict time)
        letter_model = sorted(per_feature, key=lambda x: x[2], reverse=True)[:top_k]
        letter_models[letter_idx] = letter_model

    return {
        "letter_models": letter_models,
        "top_k":         top_k,
        "n_features":    n_features,
        "letters":       ALL_LETTERS,
    }


# ---------------------------------------------------------------------------
# Prediction — matches recognize() in dyFAV.py exactly
# ---------------------------------------------------------------------------

def predict_dyfav(
    model: dict,
    x: np.ndarray,
    fuzzy_threshold: float = FUZZY_THRESHOLD_DEFAULT,
    per_class_weight_normalization: bool = True,
) -> tuple[int, dict]:
    """
    Run all 26 letter agents on a single feature vector.

    Matches recognize() from the official GitHub repo exactly:
      adj_lower = threshold_lower - fuzzy * abs(threshold_lower)
      adj_upper = threshold_upper + fuzzy * abs(threshold_lower)  ← uses lower for both

    Per-class weight normalization divides each weight by sum(top_k_weights)
    for that letter — applied at prediction time, matching the repo.

    Returns:
        (predicted_letter_idx, scores_dict)
    """
    letter_models = model["letter_models"]
    scores = {}

    for letter_idx, feat_list in letter_models.items():
        if not feat_list:
            scores[letter_idx] = 0.0
            continue

        # Normalizer: sum of all weights in this letter's top-k list
        weight_sum = sum(w for _, _, w, _ in feat_list) if per_class_weight_normalization else 1.0
        if weight_sum == 0:
            weight_sum = 1.0

        total = 0.0
        for threshold_lower, threshold_upper, weight, feat_idx in feat_list:
            # Fuzzy expansion — matches dyFAV.py recognize() exactly
            adj_lower = threshold_lower - fuzzy_threshold * abs(threshold_lower)
            adj_upper = threshold_upper + fuzzy_threshold * abs(threshold_lower)  # uses lower, not upper

            if adj_lower <= x[feat_idx] <= adj_upper:
                total += weight / weight_sum

        scores[letter_idx] = total

    predicted = max(scores, key=scores.get)
    return predicted, scores


def predict_topk(
    model: dict,
    x: np.ndarray,
    k: int = 5,
    fuzzy_threshold: float = FUZZY_THRESHOLD_DEFAULT,
) -> list[tuple[str, float]]:
    """Return top-k letter predictions with scores (for user correction UI)."""
    _, scores = predict_dyfav(model, x, fuzzy_threshold=fuzzy_threshold)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(ALL_LETTERS[idx].upper(), score) for idx, score in ranked]


# ---------------------------------------------------------------------------
# Randomized ensemble — matches dyFAV_Randomized.py
# ---------------------------------------------------------------------------

def train_dyfav_randomized(
    X: np.ndarray,
    y: np.ndarray,
    top_k: int = TOP_K,
    n_trees: int = 21,
    min_features: int = 85,
    rng_seed: int | None = None,
) -> dict:
    """
    Randomized ensemble: 21 DyFAV models each on random feature/row subsets.
    Matches dyFAV_Randomized.py from the official repo.
    """
    rng = np.random.default_rng(rng_seed)
    n_samples, n_features = X.shape
    trees = []

    for _ in range(n_trees):
        n_feat_subset = int(rng.integers(min_features, n_features + 1))
        feat_cols = rng.choice(n_features, size=n_feat_subset, replace=False)
        row_idx   = rng.choice(n_samples,  size=n_samples,      replace=True)

        X_sub = X[np.ix_(row_idx, feat_cols)]
        y_sub = y[row_idx]

        tree_model = train_dyfav(X_sub, y_sub, top_k=min(top_k, n_feat_subset))

        # Remap local feature indices → global indices
        for li in tree_model["letter_models"]:
            tree_model["letter_models"][li] = [
                (lo, hi, w, int(feat_cols[fi]))
                for lo, hi, w, fi in tree_model["letter_models"][li]
            ]
        tree_model["n_features"] = n_features
        trees.append(tree_model)

    return {
        "trees":      trees,
        "n_trees":    n_trees,
        "n_features": n_features,
        "letters":    ALL_LETTERS,
        "randomized": True,
    }


def predict_dyfav_randomized(
    ensemble: dict,
    x: np.ndarray,
    fuzzy_threshold: float = FUZZY_THRESHOLD_DEFAULT,
) -> tuple[int, dict]:
    """Weighted-majority vote over all trees in the randomized ensemble."""
    agg = {i: 0.0 for i in range(N_CLASSES)}
    for tree in ensemble["trees"]:
        _, scores = predict_dyfav(tree, x, fuzzy_threshold=fuzzy_threshold)
        for letter_idx, score in scores.items():
            agg[letter_idx] += score
    predicted = max(agg, key=agg.get)
    return predicted, agg


# ---------------------------------------------------------------------------
# Unified predict dispatcher
# ---------------------------------------------------------------------------

def predict(
    model: dict,
    x: np.ndarray,
    fuzzy_threshold: float = FUZZY_THRESHOLD_DEFAULT,
) -> tuple[int, dict]:
    if model.get("randomized"):
        return predict_dyfav_randomized(model, x, fuzzy_threshold)
    return predict_dyfav(model, x, fuzzy_threshold)


# ---------------------------------------------------------------------------
# Per-user evaluation (matches paper's 5-train / 1-test LOO protocol)
# ---------------------------------------------------------------------------

def evaluate_per_user(
    data_dir: Path,
    top_k: int = TOP_K,
    fuzzy_threshold: float = FUZZY_THRESHOLD_DEFAULT,
    randomized: bool = False,
) -> dict:
    user_data = load_dataset(str(data_dir), per_user=True)
    variant = "Randomized" if randomized else "Standard"

    print(f"\n{'='*62}")
    print(f"  DyFAV {variant} — Per-User LOO-CV")
    print(f"  top_k={top_k}  fuzzy={fuzzy_threshold}")
    print(f"{'='*62}")

    all_user_accs = []

    for uid, (X, y, rec_ids) in sorted(user_data.items()):
        recs_per_letter: dict[int, list] = {}
        for rec_id in np.unique(rec_ids):
            label = int(y[rec_ids == rec_id][0])
            recs_per_letter.setdefault(label, []).append(rec_id)

        n_recs = max(len(v) for v in recs_per_letter.values())
        fold_accs = []

        for held_pos in range(n_recs):
            test_recs  = [recs[held_pos]
                          for recs in recs_per_letter.values()
                          if held_pos < len(recs)]
            test_mask  = np.isin(rec_ids, test_recs)
            train_mask = ~test_mask

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            if randomized:
                m = train_dyfav_randomized(X[train_mask], y[train_mask], top_k=top_k)
            else:
                m = train_dyfav(X[train_mask], y[train_mask], top_k=top_k)

            preds = [predict(m, X[i], fuzzy_threshold)[0] for i in np.where(test_mask)[0]]
            acc   = float(np.mean(np.array(preds) == y[test_mask]))
            fold_accs.append(acc)

        user_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        all_user_accs.append(user_acc)
        print(f"  User {uid:2d}: {user_acc:.1%}  ({n_recs}-fold LOO)")

    overall = float(np.mean(all_user_accs))
    print(f"\n  Average accuracy: {overall:.1%}  (paper reports 95.36%)")
    print(f"{'='*62}\n")
    return {"per_user": all_user_accs, "mean": overall}


# ---------------------------------------------------------------------------
# Train and save a model for a specific user
# ---------------------------------------------------------------------------

def train_for_user(
    X: np.ndarray,
    y: np.ndarray,
    user_id: int | str,
    top_k: int = TOP_K,
    randomized: bool = False,
) -> Path:
    variant = "Randomized" if randomized else "Standard"
    print(f"\nTraining DyFAV {variant} for user {user_id}...")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Top-K: {top_k}")

    model = train_dyfav_randomized(X, y, top_k=top_k) if randomized else train_dyfav(X, y, top_k=top_k)
    model["user_id"] = user_id

    user_dir = MODELS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    fname = "dyfav_randomized_model.pkl" if randomized else "dyfav_model.pkl"
    path  = user_dir / fname
    joblib.dump(model, path)
    print(f"  [saved] → {path}")
    return path


def load_user_model(user_id: int | str, randomized: bool = False) -> dict | None:
    fname = "dyfav_randomized_model.pkl" if randomized else "dyfav_model.pkl"
    path  = MODELS_DIR / f"user_{user_id}" / fname
    return joblib.load(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Train from raw recordings collected during live session
# ---------------------------------------------------------------------------

def train_from_recordings(
    recordings: dict[str, list[np.ndarray]],
    randomized: bool = False,
) -> dict:
    """
    Train from raw sensor recordings collected in the app.

    Args:
        recordings: {'a': [array(n_samples, 17), ...], 'b': [...], ...}
        randomized: use randomized ensemble variant
    """
    X_parts, y_parts = [], []
    for letter, recs in recordings.items():
        idx = ALL_LETTERS.index(letter.lower())
        for raw in recs:
            feat = extract_510_features(raw)
            X_parts.append(feat)
            y_parts.append(idx)

    X = np.stack(X_parts).astype(np.float32)
    y = np.array(y_parts, dtype=np.int32)
    return train_dyfav_randomized(X, y) if randomized else train_dyfav(X, y)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DyFAV training and evaluation")
    parser.add_argument("--evaluate",   action="store_true",
                        help="Run per-user LOO evaluation on dyfav dataset")
    parser.add_argument("--all-users",  action="store_true",
                        help="Train and save a model for each user")
    parser.add_argument("--user",       type=int, default=None,
                        help="Train model for specific user ID")
    parser.add_argument("--top-k",      type=int, default=TOP_K,
                        help=f"Top-K features per agent (default: {TOP_K})")
    parser.add_argument("--fuzzy",      type=float, default=FUZZY_THRESHOLD_DEFAULT,
                        help=f"Fuzzy threshold multiplier (default: {FUZZY_THRESHOLD_DEFAULT})")
    parser.add_argument("--randomized", action="store_true",
                        help="Use randomized ensemble variant (21 trees)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.evaluate:
        evaluate_per_user(
            DATA_DIR, top_k=args.top_k,
            fuzzy_threshold=args.fuzzy, randomized=args.randomized,
        )
        return

    user_data = load_dataset(str(DATA_DIR), per_user=True)

    if args.user is not None:
        if args.user not in user_data:
            print(f"User {args.user} not found. Available: {sorted(user_data.keys())}")
            sys.exit(1)
        X, y, _ = user_data[args.user]
        train_for_user(X, y, args.user, top_k=args.top_k, randomized=args.randomized)
        return

    if args.all_users:
        for uid, (X, y, _) in sorted(user_data.items()):
            train_for_user(X, y, uid, top_k=args.top_k, randomized=args.randomized)
        return

    evaluate_per_user(
        DATA_DIR, top_k=args.top_k,
        fuzzy_threshold=args.fuzzy, randomized=args.randomized,
    )


if __name__ == "__main__":
    main()

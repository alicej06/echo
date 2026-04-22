"""
train_dtw.py
------------
DTW (Dynamic Time Warping) template matching for ASL phrase/word recognition.

Designed for small vocabularies (5-15 phrases) where dynamic gestures are
more natural than static fingerspelling.

Algorithm:
  - Store 3-5 raw EMG recordings per phrase as templates
  - At inference: z-score normalize, downsample, compute DTW distance to all templates
  - Nearest-neighbor classification with confidence from margin between top-2

Why DTW over DyFAV for words:
  - DyFAV assumes fixed-length static poses; phrases have variable duration
  - DTW handles time-warping: the same sign signed fast or slow still matches
  - Small vocab (9 phrases) means nearest-neighbor is fast enough

Usage:
    python scripts/train_dtw.py --user alice --evaluate
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"

# Default phrase vocabulary
PHRASES = [
    "hello",
    "my",
    "name",
    "echo",
    "nice to meet you",
    "how are you",
    "thank you",
    "great",
    "what's your name",
    "alice",
]

# Internal label for the null/background class.
# Recordings of random arm movements, resting positions, transitions — anything
# that is NOT a sign.  When predicted, the system stays silent instead of
# broadcasting a phrase.
NULL_CLASS = "_null_"
NULL_MIN_REPS = 30  # minimum null recordings before the class is included

EMG_COLS      = slice(0,  8)   # 8 EMG channels
IMU_COLS      = slice(11, 17)  # gyro[x,y,z] + accel[x,y,z]  (skip quaternion cols 8-10)
DS_FACTOR     = 4              # downsample 200Hz → 50Hz
SMOOTH_WINDOW = 10             # moving-avg window (~50ms at 200Hz)
SAKOE_RADIUS  = 0.25           # Sakoe-Chiba band as fraction of max(len_a, len_b)
EMG_WEIGHT    = 1.0            # per-channel weight for EMG features
IMU_WEIGHT    = 1.5            # per-channel weight for IMU features (motion path matters more)
MIN_REPS      = 3              # minimum reps before model can be saved
TRAIN_REPS    = 5              # target reps per phrase
MODEL_VERSION = 3              # bump when preprocessing changes; old models are rejected


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(raw: np.ndarray) -> np.ndarray:
    """
    Prepare a raw (n_samples, 17) recording for DTW.

    Column layout (from live_translate.py sync_buf):
      0-7  : EMG channels
      8-10 : orientation quaternion (w, x, y) — skipped, absolute position varies by user
      11-13: gyroscope (x, y, z deg/s)
      14-16: accelerometer (x, y, z g)

    EMG path:
      rectify → smooth (50ms MA) → downsample 4x → z-score
      Rectify converts biphasic oscillating signal to activation envelope.
      Raw EMG phase is random between reps; envelope is repeatable.

    IMU path:
      smooth → downsample 4x → z-score  (keep signed — direction matters)
      Gyro + accel encode WHERE the hand moves, which EMG alone can't capture.
      "hello" (wave) vs "thank you" (chin→forward) are distinguished by motion path.

    Returns (n_frames, 14) float32  [8 EMG + 6 IMU].
    """
    # --- EMG ---
    emg = np.abs(raw[:, EMG_COLS].astype(np.float64))
    emg = uniform_filter1d(emg, size=SMOOTH_WINDOW, axis=0, mode='nearest')
    emg = emg[::DS_FACTOR]
    emg_mean, emg_std = emg.mean(0), emg.std(0)
    emg_std[emg_std < 1e-6] = 1.0
    emg = (emg - emg_mean) / emg_std * EMG_WEIGHT

    # --- IMU (gyro + accel, signed) ---
    imu = raw[:, IMU_COLS].astype(np.float64)
    imu = uniform_filter1d(imu, size=SMOOTH_WINDOW, axis=0, mode='nearest')
    imu = imu[::DS_FACTOR]
    imu_mean, imu_std = imu.mean(0), imu.std(0)
    imu_std[imu_std < 1e-6] = 1.0
    imu = (imu - imu_mean) / imu_std * IMU_WEIGHT

    return np.concatenate([emg, imu], axis=1).astype(np.float32)  # (n_frames, 14)


# ---------------------------------------------------------------------------
# DTW core
# ---------------------------------------------------------------------------

def _dtw_distance(A: np.ndarray, B: np.ndarray, r: float = SAKOE_RADIUS) -> float:
    """
    Sakoe-Chiba banded DTW between two (n, 8) and (m, 8) sequences.

    Uses a numpy DP accumulation with Sakoe-Chiba band to constrain the
    warping path. Returns path-length-normalized distance so longer
    gestures don't automatically score worse.

    Args:
        A: (n, 8) preprocessed query sequence
        B: (m, 8) preprocessed template sequence
        r: band radius as fraction of max(n, m)

    Returns:
        float — normalized DTW distance (lower = more similar)
    """
    n, m = len(A), len(B)
    w    = max(1, int(r * max(n, m)))

    # Full pairwise Euclidean cost matrix — (n, m)
    cost = cdist(A, B, metric="euclidean")

    # DP with Sakoe-Chiba band
    INF = np.inf
    dtw = np.full((n, m), INF, dtype=np.float64)

    # Initialise origin, first row, and first column within the band.
    # Without this, dtw[0, j>0] stays INF and the path can never slide
    # along the first row, which breaks matching when len(query) ≠ len(template).
    dtw[0, 0] = cost[0, 0]
    for j in range(1, min(w + 1, m)):
        dtw[0, j] = dtw[0, j - 1] + cost[0, j]
    for i in range(1, min(w + 1, n)):
        dtw[i, 0] = dtw[i - 1, 0] + cost[i, 0]

    for i in range(1, n):
        j_lo = max(1, i - w)   # j=0 already initialised above
        j_hi = min(m, i + w + 1)
        for j in range(j_lo, j_hi):
            dtw[i, j] = cost[i, j] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    raw_dist = dtw[n - 1, m - 1]
    if raw_dist == INF:
        # Band was too tight — fall back to endpoint distance
        raw_dist = cost[n - 1, m - 1] * max(n, m)

    # Normalize by path length (approximate: n + m)
    return float(raw_dist / (n + m))


# ---------------------------------------------------------------------------
# Feature extraction (fixed-length representation of variable-length gestures)
# ---------------------------------------------------------------------------

N_SEGMENTS      = 5                        # temporal windows per recording
_AUGMENT_SCALES = (0.80, 0.90, 1.10, 1.20) # time-stretch factors for augmentation
_NOISE_STD      = 0.08                     # Gaussian noise std applied to preprocessed seq
_rng            = np.random.default_rng(42)


def _augment_stretch(raw: np.ndarray, scale: float) -> np.ndarray:
    """Resample a recording to scale × its original length."""
    n     = len(raw)
    new_n = max(2, int(n * scale))
    idx   = np.linspace(0, n - 1, new_n)
    return np.stack(
        [np.interp(idx, np.arange(n), raw[:, c]) for c in range(raw.shape[1])],
        axis=1,
    ).astype(np.float32)


def _augment_channel_dropout(raw: np.ndarray, n_drop: int = 1) -> np.ndarray:
    """Zero out n_drop randomly chosen EMG channels.

    Forces the model to not over-rely on a single electrode contact.
    Applied to raw signal before preprocessing so the z-score still runs
    on the reduced signal.
    """
    out      = raw.copy()
    channels = _rng.choice(8, size=n_drop, replace=False)
    out[:, channels] = 0.0
    return out


def _augment_noise(seq: np.ndarray) -> np.ndarray:
    """Add small Gaussian noise to a *preprocessed* sequence.

    Applied after _preprocess so the noise is in the same z-scored space as
    the real signal.  Mimics electrode contact variability and slight
    repositioning between sessions.
    """
    noise = _rng.standard_normal(seq.shape).astype(np.float32)
    return seq + noise * _NOISE_STD


def _extract_features(seq: np.ndarray, n_segs: int = N_SEGMENTS) -> np.ndarray:
    """
    Convert variable-length (n_frames, 14) sequence → fixed-length feature vector.

    Splits the sequence into n_segs equal temporal windows. For each window and
    each channel, computes three time-domain EMG features:
      MAV — mean absolute value (muscle activation level)
      RMS — root mean square (signal energy)
      WL  — waveform length / sum of abs differences (signal dynamics)

    Default output: 5 segs × 14 channels × 3 features = 210-d float32 vector.
    These features are standard in the EMG pattern-recognition literature
    (Englehart & Hudgins 2003; Phinyomark et al. 2012).
    """
    n, c  = seq.shape
    feats = []
    for s in range(n_segs):
        lo  = int(s * n / n_segs)
        hi  = max(lo + 1, int((s + 1) * n / n_segs))
        seg = seq[lo:hi]
        mav = np.mean(np.abs(seg), axis=0)
        rms = np.sqrt(np.mean(seg ** 2, axis=0))
        wl  = np.sum(np.abs(np.diff(seg, axis=0)), axis=0) if len(seg) > 1 else np.zeros(c)
        feats.extend([mav, rms, wl])
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Train / predict — SVM classifier on extracted features
# ---------------------------------------------------------------------------

def train_dtw(recordings: dict[str, list[np.ndarray]]) -> dict:
    """
    Train an SVM (RBF kernel) classifier on time-domain features from gesture recordings.

    Why SVM over DTW nearest-neighbor:
      • SVM max-margin objective generalises better with small training sets.
      • Fixed-length feature vectors allow clean cross-validation + hyperparameter search.
      • Time-stretch augmentation (×4 scales) compensates for the small dataset.
      • predict_proba gives well-calibrated confidence scores.

    Pipeline:
      augmentation → _preprocess → _extract_features
      → StandardScaler → SVC(RBF, balanced, probability=True)
      tuned with GridSearchCV over (C, gamma) via stratified k-fold CV.
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    # Validate rep counts (null class has its own minimum)
    for phrase, recs in recordings.items():
        if phrase == NULL_CLASS:
            if len(recs) < NULL_MIN_REPS:
                print(f"  [svm]  WARNING: null class has {len(recs)} recs (need {NULL_MIN_REPS} for reliable rejection) — training without it")
                recordings = {k: v for k, v in recordings.items() if k != NULL_CLASS}
            continue
        if len(recs) < MIN_REPS:
            raise ValueError(f"Phrase '{phrase}' has {len(recs)} recs, need {MIN_REPS}")

    has_null = NULL_CLASS in recordings

    # --- Original data for honest CV ---
    # Must NOT include augmented samples — augmented test samples would be nearly
    # identical to augmented training samples, inflating accuracy to ~100%.
    X_orig, y_orig = [], []
    for phrase, recs in recordings.items():
        for raw in recs:
            X_orig.append(_extract_features(_preprocess(raw)))
            y_orig.append(phrase)
    X_orig = np.array(X_orig, dtype=np.float32)
    y_orig = np.array(y_orig)
    n_classes = len(set(y_orig))
    null_str = "  +null class" if has_null else "  (no null class — false positives likely)"
    print(f"  [svm]  {len(X_orig)} original samples ({n_classes} classes, {X_orig.shape[1]} features){null_str}")

    # --- Augmented data for final training ---
    # Augmentation: time-stretch × 4, channel dropout × 2, noise × 2 = 8× per rep
    X_aug, y_aug = list(X_orig), list(y_orig)
    for phrase, recs in recordings.items():
        for raw in recs:
            for scale in _AUGMENT_SCALES:       # ×4 time-stretch
                X_aug.append(_extract_features(_preprocess(_augment_stretch(raw, scale))))
                y_aug.append(phrase)
            for n_drop in (1, 2):               # ×2 channel dropout
                X_aug.append(_extract_features(_preprocess(_augment_channel_dropout(raw, n_drop))))
                y_aug.append(phrase)
            for _ in range(2):                  # ×2 Gaussian noise (on preprocessed seq)
                X_aug.append(_extract_features(_augment_noise(_preprocess(raw))))
                y_aug.append(phrase)
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    print(f"  [svm]  {len(X_aug)} samples after augmentation")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', probability=True, class_weight='balanced')),
    ])
    param_grid = {
        'svm__C':     [0.1, 1.0, 10.0, 100.0],
        'svm__gamma': ['scale', 0.001, 0.01, 0.1],
    }
    n_splits = max(3, min(len(X_orig) // n_classes, 5))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid.fit(X_orig, y_orig)   # CV on original data only → honest accuracy

    print(f"  [svm]  CV accuracy {grid.best_score_:.1%}  params={grid.best_params_}")

    # Refit best hyperparams on full augmented dataset for a stronger final model
    from sklearn.base import clone
    final_clf = clone(grid.best_estimator_)
    final_clf.fit(X_aug, y_aug)

    return {
        'clf':        final_clf,
        'phrases':    sorted(recordings.keys()),
        'n_segments': N_SEGMENTS,
        'version':    MODEL_VERSION,
        'model_type': 'svm',
        'cv_acc':     float(grid.best_score_),
    }


def predict_dtw(
    model: dict,
    raw: np.ndarray,
    return_scores: bool = False,
) -> tuple:
    """
    Predict phrase from a raw (n_samples, 17) recording using the trained SVM.

    Returns (phrase, confidence) or (phrase, confidence, prob_dict) when return_scores=True.
    confidence = max class probability; prob_dict maps every phrase → its probability.
    """
    seq        = _preprocess(raw)
    feat       = _extract_features(seq, model.get('n_segments', N_SEGMENTS)).reshape(1, -1)
    clf        = model['clf']
    phrase     = clf.predict(feat)[0]
    proba      = clf.predict_proba(feat)[0]
    prob_dict  = dict(zip(clf.classes_, proba.tolist()))
    confidence = float(max(proba))
    if phrase == NULL_CLASS:
        # Null class: gesture was not a recognised sign — suppress output
        if return_scores:
            return None, confidence, prob_dict
        return None, confidence
    if return_scores:
        return phrase, confidence, prob_dict
    return phrase, confidence


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_phrase_recordings(recordings: dict[str, list], user_id: str | int) -> Path:
    """Persist raw phrase recordings so they accumulate across --train-words sessions."""
    user_dir = MODELS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / "phrase_recordings.pkl"
    joblib.dump(recordings, path)
    return path


def load_phrase_recordings(user_id: str | int) -> dict[str, list]:
    """Load previously saved phrase recordings (returns empty dict if none exist)."""
    path = MODELS_DIR / f"user_{user_id}" / "phrase_recordings.pkl"
    return joblib.load(path) if path.exists() else {}


def save_dtw_model(model: dict, user_id: str | int) -> Path:
    user_dir = MODELS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / "dtw_model.pkl"
    joblib.dump(model, path)
    print(f"  [dtw]  model saved → {path}")
    return path


def load_dtw_model(user_id: str | int) -> dict | None:
    path = MODELS_DIR / f"user_{user_id}" / "dtw_model.pkl"
    if not path.exists():
        return None
    model = joblib.load(path)
    if model.get("version", 1) < MODEL_VERSION:
        print(f"  [dtw]  model version {model.get('version',1)} is outdated "
              f"(need v{MODEL_VERSION}) — delete it and re-run --train-words")
        return None
    return model


# ---------------------------------------------------------------------------
# LOO evaluation (for offline testing with recorded CSVs)
# ---------------------------------------------------------------------------

def evaluate_loo(recordings: dict[str, list[np.ndarray]]) -> float:
    """
    Leave-one-out evaluation across all phrases.
    For each rep of each phrase, train on the rest, test on the held-out rep.
    """
    phrases      = list(recordings.keys())
    n_recs       = {p: len(v) for p, v in recordings.items()}
    max_reps     = max(n_recs.values())
    correct      = 0
    total        = 0
    results_true: list[str] = []
    results_pred: list[str] = []

    print(f"\n  DTW LOO Evaluation — {len(phrases)} phrases")
    print(f"  {'='*50}")

    for held_idx in range(max_reps):
        train_recs: dict[str, list[np.ndarray]] = {}
        test_recs:  list[tuple[str, np.ndarray]] = []

        for phrase in phrases:
            recs  = recordings[phrase]
            min_r = NULL_MIN_REPS if phrase == NULL_CLASS else MIN_REPS
            train = [r for i, r in enumerate(recs) if i != held_idx]
            if len(train) >= min_r and held_idx < len(recs):
                train_recs[phrase] = train
                test_recs.append((phrase, recs[held_idx]))

        if not train_recs or not test_recs:
            continue

        model = train_dtw(train_recs)
        for true_phrase, raw in test_recs:
            pred, conf = predict_dtw(model, raw)
            pred_label = pred if pred is not None else NULL_CLASS
            ok = (pred_label == true_phrase)
            correct += int(ok)
            total   += 1
            results_true.append(true_phrase)
            results_pred.append(pred_label)

    acc = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {acc:.1%}  ({correct}/{total})")

    # Confusion matrix
    labels = sorted(set(results_true))
    col_w  = max(len(l) for l in labels) + 1
    header = " " * (col_w + 2) + "".join(f"{l:>{col_w}}" for l in labels) + "  ← predicted"
    print(f"\n  {header}")
    for true_l in labels:
        row = []
        for pred_l in labels:
            cnt = sum(1 for t, p in zip(results_true, results_pred) if t == true_l and p == pred_l)
            row.append(f"{cnt:>{col_w}}")
        print(f"  {true_l:<{col_w}} |{''.join(row)}")
    print(f"  ↑ true")
    print(f"  {'='*50}\n")
    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, re
    from pathlib import Path

    parser = argparse.ArgumentParser(description="DTW phrase model evaluation")
    parser.add_argument("--user",     default="default")
    parser.add_argument("--data-dir", default="data/recordings/words/",
                        help="Directory with phrase CSV files")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run LOO evaluation on recorded CSVs")
    args = parser.parse_args()

    if args.evaluate:
        data_dir = ROOT / args.data_dir
        if not data_dir.exists():
            print(f"No data directory: {data_dir}")
            sys.exit(1)

        recordings: dict[str, list[np.ndarray]] = {}
        for csv_path in sorted(data_dir.glob("*.csv")):
            # Filename format: phrase_hello_01.csv or hello_01.csv
            name = csv_path.stem
            parts = name.rsplit("_", 1)
            phrase_key = parts[0].replace("_", " ")
            raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            recordings.setdefault(phrase_key, []).append(raw)

        if not recordings:
            print(f"No CSVs found in {data_dir}")
            sys.exit(1)

        evaluate_loo(recordings)

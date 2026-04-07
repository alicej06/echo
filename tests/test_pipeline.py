"""
Comprehensive pytest test suite for EMG-ASL pipeline.

Covers constants, signal filters, feature extraction, data loading,
synthetic data generation, all model architectures, augmentation, and
the SVM baseline classifier.

Run:
  pytest tests/ -v
Run with coverage:
  pytest tests/ --cov=src --cov-report=html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Repository root on sys.path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# TestConstants
# ===========================================================================


class TestConstants:
    def test_asl_labels_count(self):
        from src.utils.constants import ASL_LABELS
        assert len(ASL_LABELS) == 36

    def test_asl_labels_unique(self):
        from src.utils.constants import ASL_LABELS
        assert len(set(ASL_LABELS)) == len(ASL_LABELS)

    def test_sample_rate(self):
        from src.utils.constants import SAMPLE_RATE
        assert SAMPLE_RATE == 200

    def test_window_size_ms(self):
        from src.utils.constants import SAMPLE_RATE, WINDOW_SIZE_SAMPLES
        ratio = WINDOW_SIZE_SAMPLES / SAMPLE_RATE
        assert abs(ratio - 0.2) < 1e-9

    def test_feature_vector_size(self):
        from src.utils.constants import FEATURE_VECTOR_SIZE
        assert FEATURE_VECTOR_SIZE == 80


# ===========================================================================
# TestFilters
# ===========================================================================


class TestFilters:
    """Tests for src/utils/filters.py."""

    _FS = 200.0
    _N = 400   # 2 seconds at 200 Hz -- long enough for sosfiltfilt padding

    def _make_signal(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(self._N, 8)).astype(np.float64)

    def test_bandpass_shape(self):
        from src.utils.filters import bandpass_filter
        sig = self._make_signal()
        out = bandpass_filter(sig, lowcut=20.0, highcut=90.0, fs=self._FS)
        assert out.shape == sig.shape

    def test_bandpass_attenuates_dc(self):
        """A constant-level (DC) signal should be attenuated to near zero."""
        from src.utils.filters import bandpass_filter
        dc = np.ones((self._N, 8), dtype=np.float64) * 500.0
        out = bandpass_filter(dc, lowcut=20.0, highcut=90.0, fs=self._FS)
        # After bandpass the DC component should be essentially eliminated
        assert np.max(np.abs(out)) < 1.0

    def test_notch_attenuates_60hz(self):
        """Pure 60 Hz tone power should be reduced after notch filtering."""
        from src.utils.filters import notch_filter
        t = np.arange(self._N) / self._FS
        tone = np.sin(2.0 * np.pi * 60.0 * t)[:, np.newaxis] * np.ones((1, 8))
        out = notch_filter(tone.copy(), freq=60.0, fs=self._FS)
        power_in = np.mean(tone ** 2)
        power_out = np.mean(out ** 2)
        assert power_out < power_in * 0.1  # at least 10x reduction

    def test_filter_chain_no_nan(self):
        from src.utils.filters import apply_full_filter_chain
        sig = self._make_signal(seed=7)
        out = apply_full_filter_chain(
            sig,
            fs=self._FS,
            lowcut=20.0,
            highcut=90.0,   # must be < Nyquist (100 Hz at 200 Hz fs)
            notch_freq=60.0,
        )
        assert np.all(np.isfinite(out)), "Filter chain produced NaN or Inf values"


# ===========================================================================
# TestFeatureExtraction
# ===========================================================================


class TestFeatureExtraction:
    """Tests for src/utils/features.py."""

    _FS = 200.0
    _T = 40
    _C = 8

    def _make_window(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(self._T, self._C)).astype(np.float32)

    def test_extract_features_shape(self):
        from src.utils.features import extract_features
        window = self._make_window()
        feat = extract_features(window, fs=self._FS)
        assert feat.shape == (80,), f"Expected (80,), got {feat.shape}"

    def test_rms_positive(self):
        """RMS features (index 0 per channel group) must be non-negative."""
        from src.utils.features import extract_time_features
        window = self._make_window(seed=3)
        feats = extract_time_features(window)   # (5, C)
        rms_row = feats[0]
        assert np.all(rms_row >= 0.0)

    def test_features_not_nan(self):
        from src.utils.features import extract_features
        window = self._make_window(seed=5)
        feat = extract_features(window, fs=self._FS)
        assert np.all(np.isfinite(feat)), "Feature vector contains NaN or Inf"

    def test_batch_extraction(self):
        """Applying extract_features to N windows should give shape (N, 80)."""
        from src.utils.features import extract_features
        rng = np.random.default_rng(11)
        batch = rng.normal(size=(10, self._T, self._C)).astype(np.float32)
        result = np.stack(
            [extract_features(batch[i], fs=self._FS) for i in range(len(batch))],
            axis=0,
        )
        assert result.shape == (10, 80)

    def test_zero_signal_features(self):
        """An all-zero window must not crash and must return finite features."""
        from src.utils.features import extract_features
        zero_win = np.zeros((self._T, self._C), dtype=np.float32)
        feat = extract_features(zero_win, fs=self._FS)
        assert feat.shape == (80,)
        assert np.all(np.isfinite(feat))


# ===========================================================================
# TestDataLoader
# ===========================================================================


class TestDataLoader:
    """Tests for src/data/loader.py (create_windows and CSV round-trip)."""

    _N_CHANNELS = 8
    _FS = 200
    _WINDOW_SAMPLES = 40

    def _make_df(self, n_rows: int = 100, label: str = "A"):
        import pandas as pd
        rng = np.random.default_rng(0)
        data = {f"ch{i + 1}": rng.normal(size=n_rows).tolist() for i in range(self._N_CHANNELS)}
        data["timestamp_ms"] = (np.arange(n_rows) * (1000.0 / self._FS)).tolist()
        data["label"] = [label] * n_rows
        return pd.DataFrame(data)

    def test_create_windows_shape(self):
        try:
            from src.data.loader import create_windows
        except ModuleNotFoundError:
            pytest.skip("imbalanced-learn not installed")
        df = self._make_df(n_rows=100)
        windows, labels = create_windows(df, window_size=self._WINDOW_SAMPLES, overlap=0.5)
        assert windows.ndim == 3
        assert windows.shape[1] == self._WINDOW_SAMPLES
        assert windows.shape[2] == self._N_CHANNELS

    def test_create_windows_labels(self):
        try:
            from src.data.loader import create_windows
        except ModuleNotFoundError:
            pytest.skip("imbalanced-learn not installed")
        from src.utils.constants import ASL_LABELS
        df = self._make_df(n_rows=100, label="B")
        windows, labels = create_windows(df, window_size=self._WINDOW_SAMPLES, overlap=0.5)
        for lbl in labels:
            assert lbl in ASL_LABELS or lbl == "rest"

    def test_50_percent_overlap(self):
        """With 50% overlap, consecutive windows should share half their samples."""
        try:
            from src.data.loader import create_windows
        except ModuleNotFoundError:
            pytest.skip("imbalanced-learn not installed")
        df = self._make_df(n_rows=200)
        windows, labels = create_windows(df, window_size=self._WINDOW_SAMPLES, overlap=0.5)
        if len(windows) >= 2:
            # Step size with 50% overlap should be window_samples / 2 = 20
            # So window[0] and window[1] share the last 20 samples of window[0]
            # with the first 20 samples of window[1].
            step = self._WINDOW_SAMPLES // 2
            np.testing.assert_array_equal(
                windows[0, step:, :],
                windows[1, :self._WINDOW_SAMPLES - step, :],
            )

    def test_load_session_csv(self, tmp_path):
        """Save a DataFrame as CSV and reload it via load_session."""
        try:
            from src.data.loader import load_session
        except ModuleNotFoundError:
            pytest.skip("imbalanced-learn not installed")
        import pandas as pd

        df = self._make_df(n_rows=50)
        csv_path = tmp_path / "test_session.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_session(csv_path)
        assert len(loaded) == 50
        assert "label" in loaded.columns
        assert all(f"ch{i + 1}" in loaded.columns for i in range(self._N_CHANNELS))


# ===========================================================================
# TestSyntheticData
# ===========================================================================


class TestSyntheticData:
    """Tests for src/data/synthetic.py."""

    def test_generate_session_shape(self):
        from src.data.synthetic import generate_session
        df = generate_session("A", n_reps=2, fs=200)
        assert "ch1" in df.columns
        assert "label" in df.columns
        assert "timestamp_ms" in df.columns
        assert len(df) > 0

    def test_generate_session_labels(self):
        from src.data.synthetic import generate_session
        from src.utils.constants import ASL_LABELS
        df = generate_session("C", n_reps=3, fs=200)
        unique_labels = set(df["label"].unique())
        # All labels should be either the gesture label, "rest", or a known ASL label
        valid = set(ASL_LABELS) | {"rest"}
        assert unique_labels.issubset(valid)

    def test_generate_dataset_count(self, tmp_path):
        """generate_dataset should create exactly n_participants CSV files."""
        from src.data.synthetic import generate_dataset
        generate_dataset(
            output_dir=str(tmp_path),
            n_participants=3,
            n_reps=2,
            labels=["A", "B"],
        )
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 3

    def test_synthetic_snr(self):
        """The gesture signal should be stronger than baseline noise (SNR > 3 dB)."""
        from src.data.synthetic import generate_session
        df = generate_session("D", n_reps=5, fs=200, noise_level=0.1)
        gesture_rows = df[df["label"] == "D"]
        rest_rows = df[df["label"] == "rest"]
        if len(gesture_rows) == 0 or len(rest_rows) == 0:
            pytest.skip("Session contained no gesture or no rest rows")
        ch_cols = [f"ch{i + 1}" for i in range(8)]
        gesture_power = np.mean(gesture_rows[ch_cols].values ** 2)
        noise_power = np.mean(rest_rows[ch_cols].values ** 2) + 1e-12
        snr_linear = gesture_power / noise_power
        snr_db = 10.0 * np.log10(snr_linear)
        assert snr_db > 3.0, f"SNR too low: {snr_db:.2f} dB"


# ===========================================================================
# TestLSTMClassifier
# ===========================================================================


class TestLSTMClassifier:
    """Tests for ASLEMGClassifier in src/models/lstm_classifier.py."""

    def test_instantiation(self):
        from src.models.lstm_classifier import ASLEMGClassifier
        model = ASLEMGClassifier(input_size=320, hidden_size=64, num_layers=1)
        assert model is not None

    def test_forward_shape(self):
        import torch
        from src.models.lstm_classifier import ASLEMGClassifier
        model = ASLEMGClassifier(input_size=320, hidden_size=64, num_layers=1, num_classes=36)
        model.eval()
        x = torch.zeros(1, 320)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 36), f"Expected (1, 36), got {out.shape}"

    def test_predict_returns_label(self):
        from src.models.lstm_classifier import ASLEMGClassifier
        from src.utils.constants import ASL_LABELS, NUM_CLASSES
        model = ASLEMGClassifier(
            input_size=320,
            num_classes=NUM_CLASSES,
            label_names=list(ASL_LABELS),
        )
        feat = np.random.default_rng(0).normal(size=(320,)).astype(np.float32)
        label, conf = model.predict(feat)
        assert label in ASL_LABELS
        assert isinstance(conf, float)

    def test_save_load_roundtrip(self, tmp_path):
        from src.models.lstm_classifier import ASLEMGClassifier
        from src.utils.constants import ASL_LABELS, NUM_CLASSES
        model = ASLEMGClassifier(
            input_size=320,
            num_classes=NUM_CLASSES,
            label_names=list(ASL_LABELS),
        )
        feat = np.random.default_rng(1).normal(size=(320,)).astype(np.float32)
        label_before, conf_before = model.predict(feat)

        save_path = tmp_path / "lstm_test.pt"
        model.save(save_path)
        loaded = ASLEMGClassifier.load(save_path)

        label_after, conf_after = loaded.predict(feat)
        assert label_before == label_after
        assert abs(conf_before - conf_after) < 1e-5

    def test_onnx_export(self, tmp_path):
        pytest.importorskip("onnx", reason="onnx not installed")
        from src.models.lstm_classifier import ASLEMGClassifier
        model = ASLEMGClassifier(input_size=320, hidden_size=32, num_layers=1)
        onnx_path = tmp_path / "lstm_test.onnx"
        model.to_onnx(onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_confidence_range(self):
        from src.models.lstm_classifier import ASLEMGClassifier
        from src.utils.constants import ASL_LABELS, NUM_CLASSES
        model = ASLEMGClassifier(
            input_size=320,
            num_classes=NUM_CLASSES,
            label_names=list(ASL_LABELS),
        )
        rng = np.random.default_rng(2)
        for _ in range(5):
            feat = rng.normal(size=(320,)).astype(np.float32)
            _, conf = model.predict(feat)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} outside [0, 1]"


# ===========================================================================
# TestCNNLSTMClassifier
# ===========================================================================


class TestCNNLSTMClassifier:
    """Tests for CNNLSTMClassifier in src/models/cnn_lstm_classifier.py."""

    def test_instantiation(self):
        from src.models.cnn_lstm_classifier import CNNLSTMClassifier
        model = CNNLSTMClassifier(num_classes=36)
        assert model is not None

    def test_forward_shape(self):
        import torch
        from src.models.cnn_lstm_classifier import CNNLSTMClassifier
        model = CNNLSTMClassifier(num_classes=36)
        model.eval()
        x = torch.zeros(1, 40, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 36), f"Expected (1, 36), got {out.shape}"

    def test_predict_returns_label(self):
        from src.models.cnn_lstm_classifier import CNNLSTMClassifier
        from src.utils.constants import ASL_LABELS, NUM_CLASSES
        model = CNNLSTMClassifier(num_classes=NUM_CLASSES, label_names=list(ASL_LABELS))
        window = np.random.default_rng(3).normal(size=(40, 8)).astype(np.float32)
        label, conf = model.predict(window)
        assert label in ASL_LABELS

    def test_save_load_roundtrip(self, tmp_path):
        from src.models.cnn_lstm_classifier import CNNLSTMClassifier
        from src.utils.constants import ASL_LABELS, NUM_CLASSES
        model = CNNLSTMClassifier(num_classes=NUM_CLASSES, label_names=list(ASL_LABELS))
        window = np.random.default_rng(4).normal(size=(40, 8)).astype(np.float32)
        label_before, conf_before = model.predict(window)

        save_path = tmp_path / "cnn_lstm_test.pt"
        model.save(save_path)
        loaded = CNNLSTMClassifier.load(save_path)
        label_after, conf_after = loaded.predict(window)

        assert label_before == label_after
        assert abs(conf_before - conf_after) < 1e-5


# ===========================================================================
# TestConformerClassifier
# ===========================================================================


class TestConformerClassifier:
    """Tests for ConformerClassifier in src/models/conformer_classifier.py."""

    def test_instantiation(self):
        from src.models.conformer_classifier import ConformerClassifier
        model = ConformerClassifier(n_classes=36)
        assert model is not None

    def test_forward_shape(self):
        import torch
        from src.models.conformer_classifier import ConformerClassifier
        model = ConformerClassifier(n_classes=36)
        model.eval()
        x = torch.zeros(1, 40, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 36), f"Expected (1, 36), got {out.shape}"

    def test_predict_returns_label(self):
        from src.models.conformer_classifier import ConformerClassifier
        from src.utils.constants import ASL_LABELS, NUM_CLASSES
        model = ConformerClassifier(n_classes=NUM_CLASSES, label_names=list(ASL_LABELS))
        window = np.random.default_rng(5).normal(size=(40, 8)).astype(np.float32)
        labels, confs = model.predict(window)
        assert labels[0] in ASL_LABELS

    def test_onnx_export(self, tmp_path):
        pytest.importorskip("onnx", reason="onnx not installed")
        from src.models.conformer_classifier import ConformerClassifier
        model = ConformerClassifier(n_classes=36, n_layers=1, d_model=32, n_heads=4)
        onnx_path = tmp_path / "conformer_test.onnx"
        model.to_onnx(onnx_path)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_parameter_count(self):
        """Default ConformerClassifier should have more than 100K parameters."""
        from src.models.conformer_classifier import ConformerClassifier
        model = ConformerClassifier(n_classes=36)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 100_000, f"Too few parameters: {n_params:,}"


# ===========================================================================
# TestSVMClassifier
# ===========================================================================


class TestSVMClassifier:
    """Tests for SVMClassifier in src/models/svm_classifier.py."""

    _N_CLASSES = 36
    _FEATURE_SIZE = 80

    def _make_xy(self, n: int = 20, seed: int = 0):
        from src.utils.constants import ASL_LABELS
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, self._FEATURE_SIZE)).astype(np.float64)
        # Pick first n labels (or cycle) to give the SVM a multi-class problem
        labels_cycle = [ASL_LABELS[i % len(ASL_LABELS)] for i in range(n)]
        y = np.array(labels_cycle)
        return X, y

    def test_instantiation(self):
        from src.models.svm_classifier import SVMClassifier
        clf = SVMClassifier()
        assert clf is not None
        assert not clf._is_fitted

    def test_train_and_predict(self):
        from src.models.svm_classifier import SVMClassifier
        from src.utils.constants import ASL_LABELS
        clf = SVMClassifier()
        X, y = self._make_xy(n=36)  # one sample per class
        clf.fit(X, y)
        assert clf._is_fitted

        feat = np.random.default_rng(10).normal(size=(self._FEATURE_SIZE,))
        label, conf = clf.predict_proba_single(feat)
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0

    def test_predict_before_training(self):
        """Calling predict_proba_single before fit() should raise RuntimeError."""
        from src.models.svm_classifier import SVMClassifier
        clf = SVMClassifier()
        feat = np.zeros(self._FEATURE_SIZE)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba_single(feat)


# ===========================================================================
# TestAugmentation
# ===========================================================================


class TestAugmentation:
    """Tests for src/data/augmentation.py."""

    _T = 40
    _C = 8

    def _make_window(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(self._T, self._C)).astype(np.float32)

    def _make_batch(self, n: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, self._T, self._C)).astype(np.float32)

    def test_gaussian_noise_shape(self):
        from src.data.augmentation import add_gaussian_noise
        w = self._make_window()
        out = add_gaussian_noise(w, std=0.05)
        assert out.shape == w.shape

    def test_amplitude_scale_range(self):
        """Scaled values should be within 80-120% of the original."""
        from src.data.augmentation import amplitude_scale
        w = np.ones((self._T, self._C), dtype=np.float32)
        out = amplitude_scale(w, scale_range=(0.8, 1.2), per_channel=False)
        assert np.all(out >= 0.79) and np.all(out <= 1.21)

    def test_channel_dropout_max2(self):
        """channel_dropout must zero at most 2 channels."""
        from src.data.augmentation import channel_dropout
        rng = np.random.default_rng(99)
        for _ in range(20):
            w = rng.normal(size=(self._T, self._C)).astype(np.float32)
            out = channel_dropout(w, p=0.9, rng=rng)
            n_zeroed = sum(
                1 for ch in range(self._C)
                if np.allclose(out[:, ch], 0.0)
            )
            assert n_zeroed <= 2, f"Dropped {n_zeroed} channels (max is 2)"

    def test_time_shift_no_nan(self):
        from src.data.augmentation import time_shift
        w = self._make_window()
        out = time_shift(w, max_shift=5)
        assert out.shape == w.shape
        assert np.all(np.isfinite(out))

    def test_pipeline_shape(self):
        from src.data.augmentation import get_default_pipeline
        pipeline = get_default_pipeline("medium")
        batch = self._make_batch(n=8)
        out = pipeline(batch)
        assert out.shape == batch.shape

    def test_mixup_labels_sum_to_one(self):
        """Soft labels produced by mixup_emg must sum to 1.0 per row."""
        from src.data.augmentation import mixup_emg
        from src.utils.constants import ASL_LABELS
        batch = self._make_batch(n=10)
        # Use first 10 labels
        labels = np.array([ASL_LABELS[i] for i in range(10)])
        mixed_windows, soft_labels = mixup_emg(batch, labels, alpha=0.2)
        row_sums = soft_labels.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(row_sums)),
            atol=1e-5,
            err_msg="Soft labels do not sum to 1.0 per row",
        )

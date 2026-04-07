"""
Unit tests for src/models/lstm_classifier.py (ASLEMGClassifier).

Tests cover:
- forward pass shape
- predict() return type and value range
- confidence value range
- save / load round-trip
- ONNX export and validation
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.lstm_classifier import ASLEMGClassifier
from src.utils.constants import (
    ASL_LABELS,
    FEATURE_VECTOR_SIZE,
    NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

INPUT_SIZE = FEATURE_VECTOR_SIZE   # 80
HIDDEN_SIZE = 32                   # small for fast tests
NUM_LAYERS = 2
DROPOUT = 0.0                      # avoid stochastic behaviour in tests
BATCH_SIZE = 4
LABEL_NAMES = list(ASL_LABELS)     # 36 labels


# ---------------------------------------------------------------------------
# Fixture: a small deterministic model
# ---------------------------------------------------------------------------


@pytest.fixture()
def model() -> ASLEMGClassifier:
    """Return a freshly-constructed classifier in eval mode."""
    torch.manual_seed(0)
    m = ASLEMGClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        label_names=LABEL_NAMES,
    )
    m.eval()
    return m


@pytest.fixture()
def feature_vector() -> np.ndarray:
    """Return a deterministic single feature vector (shape = (INPUT_SIZE,))."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(INPUT_SIZE).astype(np.float32)


@pytest.fixture()
def batch_tensor() -> torch.Tensor:
    """Return a batch of random feature vectors (shape = (BATCH_SIZE, INPUT_SIZE))."""
    torch.manual_seed(1)
    return torch.randn(BATCH_SIZE, INPUT_SIZE)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_output_shape_2d_input(
        self, model: ASLEMGClassifier, batch_tensor: torch.Tensor
    ) -> None:
        """2-D input (batch, features) must produce logits (batch, num_classes)."""
        with torch.no_grad():
            out = model(batch_tensor)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {out.shape}"
        )

    def test_output_shape_3d_input(self, model: ASLEMGClassifier) -> None:
        """3-D input (batch, seq_len, features) must also produce (batch, num_classes)."""
        seq_len = 5
        x = torch.randn(BATCH_SIZE, seq_len, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_is_finite(
        self, model: ASLEMGClassifier, batch_tensor: torch.Tensor
    ) -> None:
        """Logits must be finite (no NaN / Inf)."""
        with torch.no_grad():
            out = model(batch_tensor)
        assert torch.all(torch.isfinite(out)), "Model produced non-finite logits"

    def test_single_sample_forward(self, model: ASLEMGClassifier) -> None:
        """batch_size=1 forward pass must not crash and output correct shape."""
        x = torch.randn(1, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_label_and_confidence(
        self,
        model: ASLEMGClassifier,
        feature_vector: np.ndarray,
    ) -> None:
        """predict() must return a (str, float) tuple."""
        result = model.predict(feature_vector)
        assert isinstance(result, tuple) and len(result) == 2
        label, conf = result
        assert isinstance(label, str), f"label should be str, got {type(label)}"
        assert isinstance(conf, float), f"confidence should be float, got {type(conf)}"

    def test_predict_label_in_vocabulary(
        self,
        model: ASLEMGClassifier,
        feature_vector: np.ndarray,
    ) -> None:
        """The predicted label must be a member of ASL_LABELS."""
        label, _ = model.predict(feature_vector)
        assert label in ASL_LABELS, f"Predicted label '{label}' not in ASL_LABELS"

    def test_predict_confidence_in_range(
        self,
        model: ASLEMGClassifier,
        feature_vector: np.ndarray,
    ) -> None:
        """Softmax confidence must be in [0, 1]."""
        _, conf = model.predict(feature_vector)
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1]"

    def test_predict_accepts_2d_input(
        self, model: ASLEMGClassifier, feature_vector: np.ndarray
    ) -> None:
        """predict() must also accept a (1, INPUT_SIZE) shaped array."""
        label, conf = model.predict(feature_vector[np.newaxis, :])
        assert label in ASL_LABELS
        assert 0.0 <= conf <= 1.0

    def test_predict_deterministic(
        self, model: ASLEMGClassifier, feature_vector: np.ndarray
    ) -> None:
        """Two consecutive predict() calls on the same input must return identical results."""
        label1, conf1 = model.predict(feature_vector)
        label2, conf2 = model.predict(feature_vector)
        assert label1 == label2
        assert conf1 == conf2

    def test_predict_without_label_names(self, feature_vector: np.ndarray) -> None:
        """Without label_names, predict() should return string integer index."""
        torch.manual_seed(0)
        m = ASLEMGClassifier(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            label_names=None,
        )
        m.eval()
        label, conf = m.predict(feature_vector)
        # label should be a string representation of an integer
        assert label.isdigit() or label.lstrip("-").isdigit(), (
            f"Expected integer string without label_names, got '{label}'"
        )
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Confidence range
# ---------------------------------------------------------------------------


class TestConfidenceRange:
    """All softmax outputs across many random inputs must be in [0, 1]."""

    def test_confidence_always_in_range(self, model: ASLEMGClassifier) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            vec = rng.standard_normal(INPUT_SIZE).astype(np.float32)
            _, conf = model.predict(vec)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} outside [0, 1]"

    def test_confidence_sums_to_one(
        self, model: ASLEMGClassifier, batch_tensor: torch.Tensor
    ) -> None:
        """Softmax probabilities over all classes must sum to 1.0."""
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        np.testing.assert_allclose(
            sums.numpy(), np.ones(BATCH_SIZE), atol=1e-5
        )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_model_save_load_predictions_match(
        self,
        model: ASLEMGClassifier,
        feature_vector: np.ndarray,
    ) -> None:
        """After save → load, predictions for the same input must be identical."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "test_model.pt"
            model.save(checkpoint_path)

            assert checkpoint_path.exists(), "Checkpoint file was not created"

            loaded = ASLEMGClassifier.load(checkpoint_path)
            loaded.eval()

            label_orig, conf_orig = model.predict(feature_vector)
            label_loaded, conf_loaded = loaded.predict(feature_vector)

            assert label_orig == label_loaded, (
                f"Labels differ after reload: '{label_orig}' vs '{label_loaded}'"
            )
            assert abs(conf_orig - conf_loaded) < 1e-6, (
                f"Confidence differs after reload: {conf_orig} vs {conf_loaded}"
            )

    def test_loaded_model_architecture_matches(
        self, model: ASLEMGClassifier
    ) -> None:
        """The loaded model must have identical architecture attributes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "arch_test.pt"
            model.save(path)
            loaded = ASLEMGClassifier.load(path)

            assert loaded.input_size == model.input_size
            assert loaded.hidden_size == model.hidden_size
            assert loaded.num_layers == model.num_layers
            assert loaded.num_classes == model.num_classes
            assert loaded.dropout_p == model.dropout_p
            assert loaded.label_names == model.label_names

    def test_checkpoint_file_created(self, model: ASLEMGClassifier) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "subdir" / "model.pt"
            model.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


class TestONNXExport:
    def test_onnx_file_created(self, model: ASLEMGClassifier) -> None:
        """to_onnx() must create a .onnx file on disk."""
        pytest.importorskip("onnx")
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            model.to_onnx(onnx_path)
            assert onnx_path.exists(), f"ONNX file not found at {onnx_path}"

    def test_onnx_model_passes_checker(self, model: ASLEMGClassifier) -> None:
        """The exported ONNX graph must pass onnx.checker.check_model."""
        onnx = pytest.importorskip("onnx")
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            model.to_onnx(onnx_path)
            onnx_model = onnx.load(str(onnx_path))
            # This raises if the model is invalid
            onnx.checker.check_model(onnx_model)

    def test_onnx_inference_matches_pytorch(
        self, model: ASLEMGClassifier, feature_vector: np.ndarray
    ) -> None:
        """ONNX Runtime output must match PyTorch output within numerical precision."""
        onnx = pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "model.onnx"
            model.to_onnx(onnx_path)

            # PyTorch reference
            x = torch.from_numpy(feature_vector[np.newaxis, :])
            with torch.no_grad():
                pt_logits = model(x).numpy()

            # ONNX Runtime
            sess = ort.InferenceSession(str(onnx_path))
            ort_inputs = {sess.get_inputs()[0].name: feature_vector[np.newaxis, :]}
            ort_logits = sess.run(None, ort_inputs)[0]

            np.testing.assert_allclose(pt_logits, ort_logits, rtol=1e-4, atol=1e-5)

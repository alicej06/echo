"""
Unit tests for src/utils/features.py

Tests validate the correctness of individual feature functions as well as
the combined extract_features pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.features import (
    extract_features,
    extract_freq_features,
    extract_time_features,
    get_feature_names,
)
from src.utils.constants import (
    N_CHANNELS,
    N_FEATURES_PER_CHANNEL,
    N_FREQ_FEATURES_PER_CHANNEL,
    N_TIME_FEATURES_PER_CHANNEL,
    SAMPLE_RATE,
    WINDOW_SIZE_SAMPLES,
    FEATURE_VECTOR_SIZE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS = float(SAMPLE_RATE)          # 200.0 Hz
T = WINDOW_SIZE_SAMPLES          # 40 samples
C = N_CHANNELS                   # 8


def _zeros() -> np.ndarray:
    return np.zeros((T, C), dtype=np.float64)


def _ones(scale: float = 1.0) -> np.ndarray:
    return np.full((T, C), scale, dtype=np.float64)


def _sine_window(freq_hz: float = 50.0) -> np.ndarray:
    t = np.arange(T) / FS
    sig = np.sin(2.0 * np.pi * freq_hz * t)
    return np.tile(sig[:, np.newaxis], (1, C)).astype(np.float64)


def _ramp_window() -> np.ndarray:
    """Linearly increasing signal per channel (monotone → no zero crossings)."""
    return np.tile(np.linspace(0, 1, T)[:, np.newaxis], (1, C))


# ---------------------------------------------------------------------------
# RMS tests
# ---------------------------------------------------------------------------


class TestRMS:
    """Tests for the RMS row (row 0) in extract_time_features output."""

    def test_rms_zero_signal(self) -> None:
        """RMS of an all-zero window must be 0 for every channel."""
        feats = extract_time_features(_zeros())
        rms_row = feats[0, :]           # shape (C,)
        np.testing.assert_array_equal(rms_row, np.zeros(C))

    def test_rms_known_value(self) -> None:
        """RMS of a constant-2.0 signal must equal 2.0 for every channel."""
        feats = extract_time_features(_ones(2.0))
        rms_row = feats[0, :]
        np.testing.assert_allclose(rms_row, np.full(C, 2.0), rtol=1e-6)

    def test_rms_sine_expected_value(self) -> None:
        """RMS of a pure sine A*sin(t) should be A/sqrt(2)."""
        amplitude = 3.0
        window = amplitude * _sine_window(40.0)
        feats = extract_time_features(window)
        rms_row = feats[0, :]
        expected = amplitude / np.sqrt(2.0)
        np.testing.assert_allclose(rms_row, expected, rtol=0.02)   # 2% tolerance

    def test_rms_positive(self) -> None:
        """RMS is always non-negative."""
        rng = np.random.default_rng(42)
        window = rng.standard_normal((T, C))
        feats = extract_time_features(window)
        assert np.all(feats[0, :] >= 0.0)


# ---------------------------------------------------------------------------
# Waveform length tests
# ---------------------------------------------------------------------------


class TestWaveformLength:
    """Tests for WL (row 2) in extract_time_features."""

    def test_waveform_length_zero_signal(self) -> None:
        """A constant signal has zero WL."""
        feats = extract_time_features(_ones(5.0))
        wl_row = feats[2, :]
        np.testing.assert_array_equal(wl_row, np.zeros(C))

    def test_waveform_length_monotonic(self) -> None:
        """Longer (more oscillatory) signal should have larger WL."""
        short_window = np.zeros((T, C))
        short_window[: T // 2, :] = 1.0  # step function — one edge

        # High-frequency oscillation → many short arcs
        long_window = _sine_window(50.0)   # at 200 Hz sampling, 50 Hz = 4 samples/period

        feats_short = extract_time_features(short_window)
        feats_long = extract_time_features(long_window)

        wl_short = feats_short[2, 0]
        wl_long = feats_long[2, 0]
        assert wl_long > wl_short, (
            f"Oscillating signal WL ({wl_long:.4f}) should exceed step WL ({wl_short:.4f})"
        )

    def test_waveform_length_positive(self) -> None:
        """WL must be non-negative for any input."""
        rng = np.random.default_rng(7)
        window = rng.standard_normal((T, C))
        feats = extract_time_features(window)
        assert np.all(feats[2, :] >= 0.0)


# ---------------------------------------------------------------------------
# Feature vector length tests
# ---------------------------------------------------------------------------


class TestFeatureVectorLength:
    """Verify the total length of combined feature vectors."""

    def test_extract_features_vector_length(self) -> None:
        """extract_features must return FEATURE_VECTOR_SIZE elements."""
        feats = extract_features(_sine_window(), fs=FS)
        assert feats.shape == (FEATURE_VECTOR_SIZE,), (
            f"Expected shape ({FEATURE_VECTOR_SIZE},), got {feats.shape}"
        )

    def test_extract_time_features_shape(self) -> None:
        """extract_time_features must return shape (N_TIME_FEATURES_PER_CHANNEL, C)."""
        feats = extract_time_features(_sine_window())
        assert feats.shape == (N_TIME_FEATURES_PER_CHANNEL, C)

    def test_extract_freq_features_shape(self) -> None:
        """extract_freq_features must return shape (N_FREQ_FEATURES_PER_CHANNEL, C)."""
        feats = extract_freq_features(_sine_window(), FS)
        assert feats.shape == (N_FREQ_FEATURES_PER_CHANNEL, C)

    def test_feature_names_length(self) -> None:
        """get_feature_names must return FEATURE_VECTOR_SIZE names."""
        names = get_feature_names(C)
        assert len(names) == FEATURE_VECTOR_SIZE

    @pytest.mark.parametrize("n_ch", [1, 4, 8])
    def test_feature_vector_varies_with_channels(self, n_ch: int) -> None:
        """Feature vector size scales linearly with channel count."""
        window = np.random.default_rng(0).standard_normal((T, n_ch))
        feats = extract_features(window, fs=FS)
        expected_len = N_FEATURES_PER_CHANNEL * n_ch
        assert feats.shape == (expected_len,)


# ---------------------------------------------------------------------------
# Frequency feature shape tests
# ---------------------------------------------------------------------------


class TestFreqFeaturesShape:
    """Verify shape and validity of frequency-domain features."""

    def test_freq_features_shape(self) -> None:
        feats = extract_freq_features(_sine_window(), FS)
        assert feats.shape == (N_FREQ_FEATURES_PER_CHANNEL, C)

    def test_freq_features_finite(self) -> None:
        """Frequency features must be finite for a clean sine input."""
        feats = extract_freq_features(_sine_window(40.0), FS)
        assert np.all(np.isfinite(feats)), "Frequency features contain NaN or Inf"

    def test_mean_freq_positive_for_sine(self) -> None:
        """Mean frequency must be positive for any non-zero signal."""
        feats = extract_freq_features(_sine_window(40.0), FS)
        mean_freq_row = feats[0, :]   # mean_freq is the first row
        assert np.all(mean_freq_row > 0.0), (
            f"Mean frequencies should be positive: {mean_freq_row}"
        )

    def test_freq_features_silent_channel_no_nan(self) -> None:
        """A zero signal must not produce NaN frequency features (eps guard)."""
        feats = extract_freq_features(_zeros(), FS)
        assert np.all(np.isfinite(feats))


# ---------------------------------------------------------------------------
# extract_features finite-output test
# ---------------------------------------------------------------------------


class TestExtractFeaturesFinite:
    """Verify that no NaN or Inf values appear in the feature output."""

    def test_extract_features_returns_finite_for_sine(self) -> None:
        feats = extract_features(_sine_window(40.0), fs=FS)
        assert np.all(np.isfinite(feats)), (
            "extract_features produced NaN or Inf values for a sine input"
        )

    def test_extract_features_returns_finite_for_random(self) -> None:
        rng = np.random.default_rng(99)
        window = rng.standard_normal((T, C))
        feats = extract_features(window, fs=FS)
        assert np.all(np.isfinite(feats)), (
            "extract_features produced NaN or Inf values for random noise input"
        )

    def test_extract_features_returns_finite_for_zeros(self) -> None:
        """Even a silent window should not produce NaN/Inf (eps guards in freq features)."""
        feats = extract_features(_zeros(), fs=FS)
        assert np.all(np.isfinite(feats))

    def test_extract_features_rejects_non_2d(self) -> None:
        """Passing a 1-D array must raise ValueError."""
        with pytest.raises(ValueError):
            extract_features(np.zeros(T), fs=FS)

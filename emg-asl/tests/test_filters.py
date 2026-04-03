"""
Unit tests for src/utils/filters.py

Tests verify that each filter function produces the expected signal-
processing effect, measured via power ratios, mean checks, and RMS values.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from src.utils.filters import (
    bandpass_filter,
    dc_remove,
    normalize_signal,
    notch_filter,
)
from src.utils.constants import N_CHANNELS, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = float(SAMPLE_RATE)          # 200.0 Hz
N_SAMPLES = 2048                 # long enough for clean PSD estimates
N_CH = N_CHANNELS                # 8


def _sine(freq_hz: float, n_samples: int = N_SAMPLES, fs: float = FS) -> np.ndarray:
    """Return a single-channel (n_samples,) sine wave at freq_hz."""
    t = np.arange(n_samples) / fs
    return np.sin(2.0 * np.pi * freq_hz * t)


def _multichannel(sig_1d: np.ndarray, n_ch: int = N_CH) -> np.ndarray:
    """Tile a 1-D signal into (n_samples, n_ch) 2-D array."""
    return np.tile(sig_1d[:, np.newaxis], (1, n_ch))


def _channel_power(sig: np.ndarray, channel: int = 0) -> float:
    """Return the total power of one channel of a 2-D signal."""
    return float(np.mean(sig[:, channel] ** 2))


def _power_db(power: float, ref_power: float) -> float:
    """Convert power ratio to decibels; guard against log(0)."""
    if power == 0.0:
        return -np.inf
    return 10.0 * np.log10(power / max(ref_power, 1e-30))


# ---------------------------------------------------------------------------
# bandpass_filter tests
# ---------------------------------------------------------------------------


class TestBandpassFilter:
    """Verify that the bandpass filter passes in-band signals and attenuates
    signals outside the 20–450 Hz passband."""

    # The EMG bandpass is 20–450 Hz at fs=200 Hz.
    # At fs=200 Hz the Nyquist is 100 Hz, so the usable range is 20–90 Hz
    # (highcut must be < Nyquist). We test attenuation at 5 Hz (below passband).
    # 500 Hz is above Nyquist so we alias it; instead we use 95 Hz which is
    # between the practical 90 Hz upper edge we pick for a 200 Hz system and
    # we demonstrate attenuation relative to a 60 Hz passband signal.

    LOWCUT = 20.0
    HIGHCUT = 90.0        # stay below Nyquist at fs=200

    def _apply(self, sig_1d: np.ndarray) -> np.ndarray:
        sig_2d = _multichannel(sig_1d)
        return bandpass_filter(sig_2d, self.LOWCUT, self.HIGHCUT, FS)

    def test_bandpass_attenuates_low_frequency(self) -> None:
        """5 Hz sine (below 20 Hz cutoff) should be significantly attenuated."""
        raw = _sine(5.0)
        raw_power = float(np.mean(raw ** 2))

        out = self._apply(raw)
        out_power = _channel_power(out)

        attenuation_db = _power_db(out_power, raw_power)
        # Expect at least -20 dB attenuation for a 4th-order filter well
        # below the passband.
        assert attenuation_db < -20.0, (
            f"Expected < -20 dB attenuation for 5 Hz signal, got {attenuation_db:.1f} dB"
        )

    def test_bandpass_passes_inband_signal(self) -> None:
        """A 50 Hz sine (within 20–90 Hz passband) should pass near-losslessly."""
        raw = _sine(50.0)
        raw_power = float(np.mean(raw ** 2))

        out = self._apply(raw)
        out_power = _channel_power(out)

        attenuation_db = _power_db(out_power, raw_power)
        # sosfiltfilt (zero-phase) has very little passband ripple for order-4
        # Butterworth; we allow -3 dB as the generous acceptance criterion.
        assert attenuation_db > -3.0, (
            f"Expected > -3 dB for in-band 50 Hz signal, got {attenuation_db:.1f} dB"
        )

    def test_bandpass_attenuates_near_nyquist(self) -> None:
        """95 Hz signal (above 90 Hz highcut) should be attenuated."""
        raw = _sine(95.0)
        raw_power = float(np.mean(raw ** 2))

        out = self._apply(raw)
        out_power = _channel_power(out)

        attenuation_db = _power_db(out_power, raw_power)
        assert attenuation_db < -3.0, (
            f"Expected < -3 dB attenuation for 95 Hz signal, got {attenuation_db:.1f} dB"
        )

    def test_bandpass_output_shape_preserved(self) -> None:
        """Output array must have identical shape to the input."""
        raw = _multichannel(_sine(50.0))
        out = bandpass_filter(raw, self.LOWCUT, self.HIGHCUT, FS)
        assert out.shape == raw.shape

    def test_bandpass_rejects_non_2d_input(self) -> None:
        """bandpass_filter must raise ValueError for 1-D input."""
        with pytest.raises(ValueError):
            bandpass_filter(_sine(50.0), self.LOWCUT, self.HIGHCUT, FS)


# ---------------------------------------------------------------------------
# notch_filter tests
# ---------------------------------------------------------------------------


class TestNotchFilter:
    """Verify that the notch filter reduces power at exactly 60 Hz."""

    NOTCH_HZ = 60.0

    def test_notch_attenuates_60hz(self) -> None:
        """60 Hz sine power should drop by > 20 dB after notch filtering."""
        raw = _sine(self.NOTCH_HZ)
        raw_2d = _multichannel(raw)
        raw_power = float(np.mean(raw ** 2))

        out = notch_filter(raw_2d, self.NOTCH_HZ, FS)
        out_power = _channel_power(out)

        attenuation_db = _power_db(out_power, raw_power)
        assert attenuation_db < -20.0, (
            f"Expected > 20 dB notch attenuation, got {attenuation_db:.1f} dB"
        )

    def test_notch_preserves_nonnotch_frequency(self) -> None:
        """A 40 Hz signal (well away from 60 Hz notch) should pass through."""
        raw = _sine(40.0)
        raw_2d = _multichannel(raw)
        raw_power = float(np.mean(raw ** 2))

        out = notch_filter(raw_2d, self.NOTCH_HZ, FS)
        out_power = _channel_power(out)

        attenuation_db = _power_db(out_power, raw_power)
        assert attenuation_db > -3.0, (
            f"40 Hz should pass; got {attenuation_db:.1f} dB"
        )

    def test_notch_output_shape_preserved(self) -> None:
        raw = _multichannel(_sine(60.0))
        out = notch_filter(raw, self.NOTCH_HZ, FS)
        assert out.shape == raw.shape

    def test_notch_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError):
            notch_filter(_sine(60.0), self.NOTCH_HZ, FS)


# ---------------------------------------------------------------------------
# dc_remove tests
# ---------------------------------------------------------------------------


class TestDCRemove:
    """Verify that dc_remove subtracts the per-channel DC mean."""

    def test_dc_remove_eliminates_offset(self) -> None:
        """After dc_remove the per-channel mean must be effectively zero."""
        dc_offset = 5.0
        raw = _multichannel(_sine(50.0) + dc_offset)

        out = dc_remove(raw)

        for ch in range(N_CH):
            residual_mean = abs(out[:, ch].mean())
            assert residual_mean < 1e-10, (
                f"Channel {ch}: mean after dc_remove = {residual_mean:.2e} (expected < 1e-10)"
            )

    def test_dc_remove_zero_mean_input_unchanged(self) -> None:
        """A zero-mean signal should be unchanged by dc_remove (within FP precision)."""
        raw = _multichannel(_sine(50.0))  # pure sine is already zero-mean
        out = dc_remove(raw)
        np.testing.assert_allclose(out, raw - raw.mean(axis=0), atol=1e-12)

    def test_dc_remove_large_offset(self) -> None:
        """Test with a large DC offset (1000 mV) to ensure numerical stability."""
        raw = _multichannel(np.ones(N_SAMPLES) * 1000.0)
        out = dc_remove(raw)
        for ch in range(N_CH):
            assert abs(out[:, ch].mean()) < 1e-10

    def test_dc_remove_output_shape_preserved(self) -> None:
        raw = _multichannel(_sine(50.0))
        out = dc_remove(raw)
        assert out.shape == raw.shape

    def test_dc_remove_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError):
            dc_remove(_sine(50.0))


# ---------------------------------------------------------------------------
# normalize_signal tests
# ---------------------------------------------------------------------------


class TestNormalizeSignal:
    """Verify RMS normalisation produces unit-RMS output per channel."""

    def test_normalize_rms_equals_one(self) -> None:
        """After normalize_signal each channel should have RMS ≈ 1.0."""
        raw = _multichannel(_sine(50.0) * 3.7)   # arbitrary amplitude
        out = normalize_signal(raw)

        for ch in range(N_CH):
            rms = float(np.sqrt(np.mean(out[:, ch] ** 2)))
            assert abs(rms - 1.0) < 1e-6, (
                f"Channel {ch}: RMS = {rms:.6f} (expected ≈ 1.0)"
            )

    def test_normalize_constant_signal(self) -> None:
        """A constant (DC) signal has well-defined RMS; output should have RMS ≈ 1."""
        raw = _multichannel(np.full(N_SAMPLES, 2.5))
        out = normalize_signal(raw)
        for ch in range(N_CH):
            rms = float(np.sqrt(np.mean(out[:, ch] ** 2)))
            assert abs(rms - 1.0) < 1e-6

    def test_normalize_silent_channel_no_crash(self) -> None:
        """A zero channel should produce finite output (eps guard)."""
        raw = np.zeros((N_SAMPLES, N_CH))
        out = normalize_signal(raw)
        assert np.all(np.isfinite(out))

    def test_normalize_output_shape_preserved(self) -> None:
        raw = _multichannel(_sine(50.0))
        out = normalize_signal(raw)
        assert out.shape == raw.shape

    def test_normalize_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError):
            normalize_signal(_sine(50.0))


# ---------------------------------------------------------------------------
# Multichannel shape preservation (all filters)
# ---------------------------------------------------------------------------


class TestMultichannelShapePreservation:
    """All filter functions must return exactly the same shape as their input."""

    @pytest.fixture()
    def sample_signal(self) -> np.ndarray:
        return _multichannel(_sine(50.0))

    def test_bandpass_shape(self, sample_signal: np.ndarray) -> None:
        out = bandpass_filter(sample_signal, 20.0, 90.0, FS)
        assert out.shape == sample_signal.shape

    def test_notch_shape(self, sample_signal: np.ndarray) -> None:
        out = notch_filter(sample_signal, 60.0, FS)
        assert out.shape == sample_signal.shape

    def test_dc_remove_shape(self, sample_signal: np.ndarray) -> None:
        out = dc_remove(sample_signal)
        assert out.shape == sample_signal.shape

    def test_normalize_shape(self, sample_signal: np.ndarray) -> None:
        out = normalize_signal(sample_signal)
        assert out.shape == sample_signal.shape

    @pytest.mark.parametrize("n_ch", [1, 4, 8, 16])
    def test_arbitrary_channel_counts(self, n_ch: int) -> None:
        """Shape must be preserved for non-standard channel counts."""
        sig = np.random.default_rng(0).standard_normal((N_SAMPLES, n_ch))
        out = dc_remove(sig)
        assert out.shape == sig.shape

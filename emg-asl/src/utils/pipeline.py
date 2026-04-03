"""
EMG ingestion and windowing pipeline.

``EMGPipeline`` is the single entry point for raw BLE data.  It:

1. Parses raw bytes arriving from BLE (8-channel interleaved int16, little-endian).
2. Pushes decoded samples into a circular ring buffer.
3. Slices windows (``WINDOW_SIZE_SAMPLES`` long, ``STEP_SIZE_SAMPLES`` apart).
4. Applies the full filter chain (DC removal → bandpass → notch → RMS normalise).
5. Extracts the flat feature vector for downstream inference.

Typical usage
-------------
::

    pipeline = EMGPipeline()
    pipeline.ingest_bytes(ble_payload)          # call each time BLE chunk arrives
    while True:
        window = pipeline.get_next_window()     # returns None when buffer is empty
        if window is None:
            break
        features = pipeline.process_window(window)  # shape (FEATURE_VECTOR_SIZE,)
        # pass features to classifier …
"""

from __future__ import annotations

import struct
import threading
from collections import deque
from typing import Optional

import numpy as np

from .constants import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    N_CHANNELS,
    NOTCH_FREQ,
    SAMPLE_RATE,
    STEP_SIZE_SAMPLES,
    WINDOW_SIZE_SAMPLES,
)
from .features import extract_features
from .filters import apply_full_filter_chain


# ---------------------------------------------------------------------------
# EMGPipeline
# ---------------------------------------------------------------------------


class EMGPipeline:
    """Thread-safe EMG ingestion, windowing, filtering and feature-extraction pipeline.

    Parameters
    ----------
    n_channels:
        Number of EMG channels expected in the BLE stream.
    sample_rate:
        Sampling frequency in Hz.
    window_size_samples:
        Number of samples per analysis window.
    step_size_samples:
        Number of new samples required before a new window is emitted
        (determines overlap).
    bandpass_low, bandpass_high:
        Bandpass filter corner frequencies in Hz.
    notch_freq:
        Power-line notch frequency in Hz.
    """

    _BYTES_PER_SAMPLE: int = 2  # int16
    _SAMPLE_STRUCT: str = "<h"  # little-endian signed 16-bit integer

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        sample_rate: float = SAMPLE_RATE,
        window_size_samples: int = WINDOW_SIZE_SAMPLES,
        step_size_samples: int = STEP_SIZE_SAMPLES,
        bandpass_low: float = BANDPASS_LOW,
        bandpass_high: float = BANDPASS_HIGH,
        notch_freq: float = NOTCH_FREQ,
    ) -> None:
        self.n_channels = n_channels
        self.sample_rate = float(sample_rate)
        self.window_size = window_size_samples
        self.step_size = step_size_samples
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq

        # Ring buffer: each entry is a 1-D array of length n_channels
        # deque with maxlen acts as a circular buffer automatically.
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size_samples * 4)

        # Counts how many new samples have arrived since the last window was yielded.
        self._samples_since_last_window: int = 0

        # Pending windows ready to be consumed by get_next_window().
        self._window_queue: deque[np.ndarray] = deque()

        # Partial-frame accumulator for BLE payloads that don't align on sample boundaries.
        self._partial_bytes: bytes = b""

        # Thread safety — ingest_bytes may be called from a BLE callback thread.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_bytes(self, raw_bytes: bytes) -> int:
        """Parse raw BLE bytes and push decoded samples into the ring buffer.

        The BLE payload is expected to be 8-channel interleaved int16 values
        in little-endian byte order::

            [ch0_s0_lo, ch0_s0_hi, ch1_s0_lo, ch1_s0_hi, ..., ch7_s0_hi,
             ch0_s1_lo, ch0_s1_hi, ..., ch7_s1_hi, ...]

        Parameters
        ----------
        raw_bytes:
            Raw payload bytes from BLE notification.

        Returns
        -------
        int
            Number of complete samples (rows) decoded and buffered.
        """
        with self._lock:
            data = self._partial_bytes + raw_bytes
            bytes_per_frame = self.n_channels * self._BYTES_PER_SAMPLE
            n_complete_frames = len(data) // bytes_per_frame
            consumed = n_complete_frames * bytes_per_frame

            if n_complete_frames == 0:
                self._partial_bytes = data
                return 0

            # Decode all complete frames at once using numpy for speed.
            frame_bytes = data[:consumed]
            self._partial_bytes = data[consumed:]

            # Parse as int16 array, then reshape to (n_frames, n_channels).
            raw_int16 = np.frombuffer(frame_bytes, dtype="<i2").reshape(
                n_complete_frames, self.n_channels
            )
            samples = raw_int16.astype(np.float64)

            for row in samples:
                self._buffer.append(row.copy())
                self._samples_since_last_window += 1

                # Emit a window whenever we have accumulated enough new samples.
                if (
                    self._samples_since_last_window >= self.step_size
                    and len(self._buffer) >= self.window_size
                ):
                    window = np.array(list(self._buffer)[-self.window_size :])
                    self._window_queue.append(window)
                    self._samples_since_last_window = 0

            return n_complete_frames

    def get_next_window(self) -> Optional[np.ndarray]:
        """Return the next pending EMG window or ``None`` if none is ready.

        Returns
        -------
        np.ndarray or None
            Raw (unfiltered) window of shape ``(window_size_samples, n_channels)``,
            dtype float64; or ``None`` when the internal queue is empty.
        """
        with self._lock:
            if self._window_queue:
                return self._window_queue.popleft()
            return None

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Apply the full signal-processing chain and extract features.

        This method is deliberately *not* protected by ``_lock`` because it
        is computationally intensive and can safely run on a worker thread
        while ``ingest_bytes`` fills the buffer concurrently.

        Parameters
        ----------
        window:
            Raw EMG window, shape ``(window_size_samples, n_channels)``.

        Returns
        -------
        np.ndarray
            Flat feature vector, shape ``(10 * n_channels,)``.
        """
        filtered = apply_full_filter_chain(
            window,
            fs=self.sample_rate,
            lowcut=self.bandpass_low,
            highcut=self.bandpass_high,
            notch_freq=self.notch_freq,
        )
        return extract_features(filtered, fs=self.sample_rate)

    def reset(self) -> None:
        """Clear all internal state.

        Call this at the start of a new calibration session or after a
        prolonged disconnection to prevent stale samples from polluting
        the next window.
        """
        with self._lock:
            self._buffer.clear()
            self._window_queue.clear()
            self._partial_bytes = b""
            self._samples_since_last_window = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buffer_length(self) -> int:
        """Number of samples currently in the ring buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def pending_windows(self) -> int:
        """Number of windows waiting to be consumed by ``get_next_window()``."""
        with self._lock:
            return len(self._window_queue)

    @property
    def feature_vector_size(self) -> int:
        """Expected length of the feature vector produced by ``process_window``."""
        return 10 * self.n_channels

    # ------------------------------------------------------------------
    # Context-manager support (for use in ``async with`` blocks)
    # ------------------------------------------------------------------

    def __enter__(self) -> "EMGPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EMGPipeline("
            f"n_channels={self.n_channels}, "
            f"fs={self.sample_rate}, "
            f"window={self.window_size}, "
            f"step={self.step_size}, "
            f"buffer_length={self.buffer_length}, "
            f"pending_windows={self.pending_windows}"
            f")"
        )

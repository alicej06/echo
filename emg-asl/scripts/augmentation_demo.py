"""
Augmentation demo: visualize each EMG augmentation on a synthetic window.

Generates one synthetic forearm sEMG window, applies each augmentation
individually, then plots a 3x3 grid (original + 8 augmented versions).

Usage:
    python scripts/augmentation_demo.py

Output:
    docs/augmentation_demo.png
"""

from __future__ import annotations

import os
import sys

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- no display required
import matplotlib.pyplot as plt

from src.utils.constants import N_CHANNELS, SAMPLE_RATE, WINDOW_SIZE_SAMPLES
from src.data.augmentation import (
    add_gaussian_noise,
    amplitude_scale,
    time_warp,
    channel_dropout,
    time_shift,
    magnitude_warp,
    band_stop_noise,
    electrode_offset,
)


# ---------------------------------------------------------------------------
# Synthetic EMG window generator
# ---------------------------------------------------------------------------


def make_synthetic_window(
    T: int = WINDOW_SIZE_SAMPLES,
    C: int = N_CHANNELS,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a single synthetic forearm sEMG window of shape (T, C).

    The window mimics a typical active EMG burst: bandlimited noise with a
    smooth Gaussian amplitude envelope, different carrier frequencies and
    phases per channel so the plot is visually interesting.

    Parameters
    ----------
    T:
        Number of time samples.
    C:
        Number of channels.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (T, C), float32
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T / SAMPLE_RATE, T)  # time axis in seconds

    # Gaussian burst envelope (peaks in the middle of the window)
    envelope = np.exp(-((t - t.mean()) ** 2) / (2 * (0.04) ** 2))

    window = np.empty((T, C), dtype=np.float32)
    for c in range(C):
        freq = rng.uniform(30.0, 120.0)   # carrier frequency per channel
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.4, 1.0)
        carrier = amplitude * np.sin(2 * np.pi * freq * t + phase)
        # Bandlimited noise modulated by envelope
        noise = rng.standard_normal(T) * 0.15
        window[:, c] = (carrier * envelope + noise * envelope).astype(np.float32)

    return window


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_window(ax: plt.Axes, window: np.ndarray, title: str) -> None:
    """
    Plot all channels of an (T, C) EMG window on a single axes, offset
    vertically so channels do not overlap.
    """
    T, C = window.shape
    t = np.arange(T)
    offsets = np.arange(C) * 2.0  # vertical spacing between channels

    for c in range(C):
        ax.plot(
            t,
            window[:, c] + offsets[c],
            lw=0.8,
            color=plt.cm.tab10(c / C),
            alpha=0.85,
        )

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(42)
    window = make_synthetic_window(seed=42)

    # Each entry: (title, augmented_window)
    panels = [
        ("Original", window.copy()),
        (
            "add_gaussian_noise\n(std=0.05)",
            add_gaussian_noise(window, std=0.05, rng=rng),
        ),
        (
            "amplitude_scale\n(0.8-1.2, per_channel)",
            amplitude_scale(window, scale_range=(0.8, 1.2), per_channel=True, rng=rng),
        ),
        (
            "time_warp\n(sigma=0.1, knot=4)",
            time_warp(window, sigma=0.1, knot=4, rng=rng),
        ),
        (
            "channel_dropout\n(p=0.25)",
            channel_dropout(window, p=0.25, rng=rng),
        ),
        (
            "time_shift\n(max_shift=5)",
            time_shift(window, max_shift=5, rng=rng),
        ),
        (
            "magnitude_warp\n(sigma=0.15, knot=4)",
            magnitude_warp(window, sigma=0.15, knot=4, rng=rng),
        ),
        (
            "band_stop_noise\n(60 Hz interference)",
            band_stop_noise(window, center_hz=60.0, bandwidth_hz=4.0, rng=rng),
        ),
        (
            "electrode_offset\n(+/-10% DC drift)",
            electrode_offset(window, offset_range=(-0.1, 0.1), rng=rng),
        ),
    ]

    assert len(panels) == 9, "Expected 9 panels (original + 8 augmentations)"

    fig, axes = plt.subplots(
        3, 3,
        figsize=(12, 8),
        facecolor="#f8f8f8",
    )
    fig.suptitle(
        "EMG-ASL Data Augmentation Techniques\n"
        f"Synthetic sEMG window: {WINDOW_SIZE_SAMPLES} samples x {N_CHANNELS} channels "
        f"@ {SAMPLE_RATE} Hz",
        fontsize=11,
        y=0.98,
    )

    for ax, (title, w) in zip(axes.flat, panels):
        _plot_window(ax, w, title)

    # Highlight original panel with a subtle border
    for spine in axes[0, 0].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.6)

    # Add a channel legend below the figure
    handles = [
        matplotlib.patches.Patch(
            color=plt.cm.tab10(c / N_CHANNELS),
            label=f"ch{c + 1}",
        )
        for c in range(N_CHANNELS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=N_CHANNELS,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Ensure output directory exists
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "augmentation_demo.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(out_path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()

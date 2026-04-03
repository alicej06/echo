#!/usr/bin/env python3
"""
validate_session.py — Signal quality validator for EMG-ASL session CSVs.

Inspects one or more session CSV files for common data quality issues before
training.  Detects bad electrode contact, ADC saturation, disconnected channels,
power-line interference, missing data, class imbalance, and estimates per-channel
signal-to-noise ratio.

Usage
-----
    # Validate a single session file
    python scripts/validate_session.py data/raw/P001_20260227_000000.csv

    # Validate all CSV files in a directory
    python scripts/validate_session.py data/raw/

    # Validate and auto-fix minor issues (saves {filename}_fixed.csv)
    python scripts/validate_session.py data/raw/P001_20260227_000000.csv --fix

    # Validate a directory and fix all files
    python scripts/validate_session.py data/raw/ --fix

    # Adjust minimum windows per class threshold
    python scripts/validate_session.py data/raw/ --min-windows 10

    # Use strict clipping threshold
    python scripts/validate_session.py data/raw/ --clip-threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the script is run directly.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.constants import (  # noqa: E402
    N_CHANNELS,
    NOTCH_FREQ,
    SAMPLE_RATE,
    WINDOW_SIZE_MS,
    WINDOW_SIZE_SAMPLES,
    OVERLAP,
)

# ---------------------------------------------------------------------------
# Quality check thresholds (can be overridden via CLI)
# ---------------------------------------------------------------------------

_DEFAULT_CLIP_THRESHOLD_PCT: float = 1.0    # warn if >1% of samples at ADC limits
_DEFAULT_ADC_MIN: float = 0.0               # 12-bit ADC floor
_DEFAULT_ADC_MAX: float = 4095.0            # 12-bit ADC ceiling

_DEFAULT_RMS_LOW_UV: float = 10.0           # µV — warn if RMS below this (bad contact)
_DEFAULT_RMS_HIGH_UV: float = 1000.0        # µV — warn if RMS above this (saturation)

_DEFAULT_FLAT_STD_THRESHOLD: float = 1e-3   # std dev below this → channel likely disconnected

_DEFAULT_POWERLINE_RATIO: float = 3.0       # 60 Hz component must be this many × avg spectral power
_DEFAULT_MIN_WINDOWS: int = 5               # minimum windows per class; warn if below

_STEP_SIZE_SAMPLES: int = max(1, int(WINDOW_SIZE_SAMPLES * (1.0 - OVERLAP)))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChannelResult:
    """Quality assessment for a single EMG channel."""
    channel: str
    status: str               # "GOOD", "WARNING", or "ERROR"
    rms_uv: float
    snr_db: float
    clipping_pct: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class SessionReport:
    """Full quality report for one session CSV."""
    filepath: Path
    n_samples: int
    n_labels: int
    duration_s: float
    channel_results: List[ChannelResult]
    label_window_counts: dict[str, int]
    missing_data_issues: List[str]
    overall_status: str        # "PASS", "PASS with N warnings", or "FAIL"
    n_warnings: int
    n_errors: int


# ---------------------------------------------------------------------------
# Per-channel checks
# ---------------------------------------------------------------------------


def _check_clipping(
    ch_data: np.ndarray,
    threshold_pct: float,
    adc_min: float,
    adc_max: float,
) -> Tuple[float, Optional[str]]:
    """
    Check for ADC clipping (samples stuck at min or max value).

    Parameters
    ----------
    ch_data:
        1-D array of raw or normalised channel samples.
    threshold_pct:
        Percentage above which clipping is considered a warning.
    adc_min, adc_max:
        ADC floor and ceiling values.  If data has been normalised away from
        12-bit range, clipping is inferred from values within 0.1% of the
        observed min/max.

    Returns
    -------
    (clipping_pct, warning_message_or_None)
    """
    n = len(ch_data)
    if n == 0:
        return 0.0, None

    # Determine whether data is in raw 12-bit range or has been normalised.
    # Raw 12-bit data has values in [0, 4095]; normalised data typically lives
    # in [-10, 10] (after z-score) or [-1, 1] (after min-max).
    data_max = float(ch_data.max())
    data_min = float(ch_data.min())
    data_range = data_max - data_min

    if data_range == 0:
        # Completely flat channel — flag as 100% clipped for visibility
        return 100.0, "clipping 100.0% — check ADC saturation or electrode gel"

    if data_max > 10.0 or data_min < -10.0:
        # Likely raw ADC values (e.g. 12-bit 0..4095 or bipolar mV)
        n_clipped = int(np.sum(ch_data <= adc_min) + np.sum(ch_data >= adc_max))
    else:
        # Likely normalised (z-score or min-max).  Clipping in this case means
        # many samples share the *exact* global min or max value because the
        # amplifier/ADC saturated before normalisation.  We count samples within
        # a tight tolerance of the observed extremes (0.1% of the data range).
        tol = data_range * 0.001
        n_clipped = int(
            np.sum(ch_data <= data_min + tol) + np.sum(ch_data >= data_max - tol)
        )
        # Subtract the single min and max samples themselves (not clipping)
        n_clipped = max(0, n_clipped - 2)

    pct = 100.0 * n_clipped / n
    if pct > threshold_pct:
        msg = f"clipping {pct:.1f}% — check ADC saturation or electrode gel"
        return pct, msg
    return pct, None


def _check_rms(
    ch_data: np.ndarray,
    rms_low: float,
    rms_high: float,
) -> Tuple[float, Optional[str]]:
    """
    Check that RMS power is within the expected sEMG range.

    If the signal has been z-score normalised (mean≈0, std≈1), a direct µV
    comparison is not possible; the check is skipped for normalised data and
    RMS is reported in normalised units instead.

    Returns
    -------
    (rms_value, warning_message_or_None)
    """
    rms = float(np.sqrt(np.mean(ch_data ** 2)))
    data_std = float(np.std(ch_data))

    # Detect if likely z-score normalised: std close to 1
    is_normalised = 0.5 < data_std < 2.0 and abs(np.mean(ch_data)) < 0.5

    if is_normalised:
        # Cannot compare to µV thresholds; report relative RMS
        if rms < 0.05:
            return rms, f"very low RMS={rms:.4f} (normalised) — check electrode contact"
        return rms, None

    if rms < rms_low:
        return rms, f"low signal RMS={rms:.1f}µV — check electrode contact"
    if rms > rms_high:
        return rms, f"high signal RMS={rms:.1f}µV — possible saturation"
    return rms, None


def _check_flat_signal(
    ch_data: np.ndarray,
    std_threshold: float,
) -> Optional[str]:
    """
    Detect flat/constant signals that indicate a disconnected electrode.

    Returns a warning string if the channel's standard deviation is below
    ``std_threshold``.
    """
    std = float(np.std(ch_data))
    if std < std_threshold:
        return f"flat signal (std={std:.2e}) — channel may be disconnected"
    return None


def _check_powerline_noise(
    ch_data: np.ndarray,
    fs: int,
    notch_freq: float,
    power_ratio_threshold: float,
) -> Optional[str]:
    """
    Detect 60 Hz (or configured notch frequency) power-line interference via FFT.

    Computes the one-sided power spectrum and compares the spectral power in a
    narrow band around ``notch_freq`` (±2 Hz) to the mean spectral power of
    the rest of the spectrum.  Warns if the ratio exceeds ``power_ratio_threshold``.

    Returns
    -------
    str or None
        Warning message if power-line interference is detected.
    """
    n = len(ch_data)
    if n < fs:  # Need at least 1 second of data for a meaningful FFT
        return None

    # Zero-mean the signal to suppress DC offset
    signal = ch_data - ch_data.mean()

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(signal))
    power = fft_mag ** 2

    # Bin indices for the notch frequency band (±2 Hz)
    band_mask = (freqs >= notch_freq - 2) & (freqs <= notch_freq + 2)
    background_mask = ~band_mask & (freqs > 5)  # exclude DC and the band itself

    if not np.any(band_mask) or not np.any(background_mask):
        return None

    notch_power = float(np.mean(power[band_mask]))
    background_power = float(np.mean(power[background_mask]))

    if background_power == 0:
        return None

    ratio = notch_power / background_power
    if ratio > power_ratio_threshold:
        return (
            f"power-line noise at {notch_freq:.0f} Hz "
            f"({ratio:.1f}× background) — consider applying notch filter"
        )
    return None


def _estimate_snr(ch_data: np.ndarray, fs: int) -> float:
    """
    Estimate signal-to-noise ratio in dB for a single EMG channel.

    Strategy: treat the signal energy in the sEMG band (20–450 Hz) as
    "signal" and the energy outside that band (below 20 Hz and above 450 Hz,
    up to Nyquist) as "noise".  This is a heuristic; true SNR requires a
    controlled noise measurement.

    Returns
    -------
    float
        Estimated SNR in dB.  Returns 0.0 if insufficient data.
    """
    n = len(ch_data)
    if n < fs:
        return 0.0

    signal = ch_data - ch_data.mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2

    semg_band = (freqs >= 20) & (freqs <= 450)
    noise_band = ~semg_band

    signal_power = float(np.mean(power[semg_band])) if np.any(semg_band) else 0.0
    noise_power = float(np.mean(power[noise_band])) if np.any(noise_band) else 0.0

    if noise_power == 0 or signal_power == 0:
        return 0.0

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return round(snr_db, 1)


# ---------------------------------------------------------------------------
# Missing data checks
# ---------------------------------------------------------------------------


def _check_missing_data(df: pd.DataFrame, channel_cols: List[str]) -> List[str]:
    """
    Check for NaN, inf, and timestamp gaps.

    Returns a list of issue description strings (empty list if clean).
    """
    issues: List[str] = []

    # NaN check
    nan_counts = df[channel_cols + ["label"]].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        for col, cnt in nan_cols.items():
            issues.append(f"NaN values in column '{col}': {cnt} rows")

    # Inf check
    for col in channel_cols:
        n_inf = int(np.sum(~np.isfinite(df[col].values)))
        if n_inf > 0:
            issues.append(f"Inf values in column '{col}': {n_inf} rows")

    # Timestamp gap check (detect jumps > 3× expected interval)
    if "timestamp_ms" in df.columns:
        ts = df["timestamp_ms"].values
        if len(ts) > 1:
            diffs = np.diff(ts)
            expected_dt = 1000.0 / SAMPLE_RATE  # ms
            large_gaps = np.sum(diffs > expected_dt * 3)
            if large_gaps > 0:
                max_gap_ms = float(np.max(diffs))
                issues.append(
                    f"Timestamp gaps detected: {large_gaps} gaps "
                    f"(largest={max_gap_ms:.1f}ms, expected≤{expected_dt * 3:.1f}ms) "
                    "— possible data dropout"
                )

    return issues


# ---------------------------------------------------------------------------
# Class balance check
# ---------------------------------------------------------------------------


def _count_windows_per_label(
    df: pd.DataFrame,
    channel_cols: List[str],
    window_samples: int = WINDOW_SIZE_SAMPLES,
    step_samples: int = _STEP_SIZE_SAMPLES,
) -> dict[str, int]:
    """
    Count how many fully-labelled windows exist for each class.

    A window is counted for label L if every sample in the window has label L
    (same definition used by create_windows in loader.py).

    Returns
    -------
    dict mapping label string → window count
    """
    label_arr = df["label"].values
    counts: dict[str, int] = {}

    for start in range(0, len(label_arr) - window_samples + 1, step_samples):
        end = start + window_samples
        window_labels = label_arr[start:end]
        unique = set(window_labels)
        if len(unique) != 1:
            continue
        lbl = next(iter(unique))
        if not lbl or str(lbl) == "nan":
            continue
        counts[str(lbl)] = counts.get(str(lbl), 0) + 1

    return counts


# ---------------------------------------------------------------------------
# Core validation function
# ---------------------------------------------------------------------------


def validate_session(
    filepath: str | Path,
    clip_threshold_pct: float = _DEFAULT_CLIP_THRESHOLD_PCT,
    adc_min: float = _DEFAULT_ADC_MIN,
    adc_max: float = _DEFAULT_ADC_MAX,
    rms_low_uv: float = _DEFAULT_RMS_LOW_UV,
    rms_high_uv: float = _DEFAULT_RMS_HIGH_UV,
    flat_std_threshold: float = _DEFAULT_FLAT_STD_THRESHOLD,
    powerline_ratio: float = _DEFAULT_POWERLINE_RATIO,
    min_windows: int = _DEFAULT_MIN_WINDOWS,
) -> SessionReport:
    """
    Run all signal quality checks on a single session CSV.

    Parameters
    ----------
    filepath:
        Path to the session CSV file.
    clip_threshold_pct:
        Percentage of samples that may be at ADC limits before a clipping
        warning is raised.
    adc_min, adc_max:
        ADC floor and ceiling (0 and 4095 for 12-bit DAQ).
    rms_low_uv, rms_high_uv:
        RMS power bounds in µV.  Raw (non-normalised) data only.
    flat_std_threshold:
        Standard deviation below which a channel is considered flat/disconnected.
    powerline_ratio:
        Ratio of 60 Hz spectral power to background power above which a
        power-line noise warning is issued.
    min_windows:
        Minimum number of full windows required per label class.

    Returns
    -------
    SessionReport
        Structured quality report for the session.
    """
    path = Path(filepath)

    # Load — gracefully handle missing required columns
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Cannot read '{path}': {exc}") from exc

    channel_cols = [f"ch{i + 1}" for i in range(N_CHANNELS)]
    missing_cols = [c for c in channel_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Session file '{path.name}' is missing channel columns: {missing_cols}. "
            "Expected ch1..ch8."
        )
    if "label" not in df.columns:
        raise ValueError(f"Session file '{path.name}' has no 'label' column.")

    # Basic stats
    n_samples = len(df)
    n_labels = df["label"].nunique()
    duration_s = n_samples / SAMPLE_RATE

    # Missing data checks
    missing_issues = _check_missing_data(df, channel_cols)

    # Per-channel checks
    channel_results: List[ChannelResult] = []
    for ch in channel_cols:
        ch_data = pd.to_numeric(df[ch], errors="coerce").fillna(0.0).values.astype(np.float32)
        ch_warnings: List[str] = []

        # 1. Clipping
        clip_pct, clip_warn = _check_clipping(ch_data, clip_threshold_pct, adc_min, adc_max)
        if clip_warn:
            ch_warnings.append(clip_warn)

        # 2. Flat signal (disconnected)
        flat_warn = _check_flat_signal(ch_data, flat_std_threshold)
        if flat_warn:
            ch_warnings.append(flat_warn)

        # 3. RMS power
        rms_val, rms_warn = _check_rms(ch_data, rms_low_uv, rms_high_uv)
        if rms_warn:
            ch_warnings.append(rms_warn)

        # 4. Power-line noise
        pl_warn = _check_powerline_noise(ch_data, SAMPLE_RATE, NOTCH_FREQ, powerline_ratio)
        if pl_warn:
            ch_warnings.append(pl_warn)

        # 5. SNR estimate
        snr_db = _estimate_snr(ch_data, SAMPLE_RATE)

        status = "GOOD" if not ch_warnings else "WARNING"

        # Format RMS for display (use normalised units if data is normalised)
        data_std = float(np.std(ch_data))
        is_normalised = 0.5 < data_std < 2.0 and abs(float(np.mean(ch_data))) < 0.5
        rms_display = rms_val if not is_normalised else rms_val

        channel_results.append(
            ChannelResult(
                channel=ch,
                status=status,
                rms_uv=round(rms_display, 1),
                snr_db=snr_db,
                clipping_pct=round(clip_pct, 2),
                warnings=ch_warnings,
            )
        )

    # Class balance: count windows per label
    label_window_counts = _count_windows_per_label(df, channel_cols)
    low_class_warnings: List[str] = []
    for lbl, cnt in label_window_counts.items():
        if cnt < min_windows:
            low_class_warnings.append(
                f"{lbl} only has {cnt} window(s) (minimum: {min_windows})"
            )

    # Tally totals
    n_warnings = (
        sum(1 for cr in channel_results if cr.status == "WARNING")
        + len(missing_issues)
        + len(low_class_warnings)
    )
    n_errors = sum(1 for cr in channel_results if cr.status == "ERROR")

    if n_errors > 0:
        overall = "FAIL"
    elif n_warnings > 0:
        overall = f"PASS with {n_warnings} warning{'s' if n_warnings > 1 else ''}"
    else:
        overall = "PASS"

    return SessionReport(
        filepath=path,
        n_samples=n_samples,
        n_labels=n_labels,
        duration_s=round(duration_s, 1),
        channel_results=channel_results,
        label_window_counts=label_window_counts,
        missing_data_issues=missing_issues,
        overall_status=overall,
        n_warnings=n_warnings,
        n_errors=n_errors,
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_report(report: SessionReport) -> str:
    """
    Format a SessionReport as a human-readable quality report string.

    Parameters
    ----------
    report:
        SessionReport produced by :func:`validate_session`.

    Returns
    -------
    str
        Multi-line formatted report.
    """
    lines: List[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"Session Quality Report: {report.filepath.name}")
    lines.append(sep)
    lines.append(
        f"Samples: {report.n_samples:,}  |  "
        f"Labels: {report.n_labels}  |  "
        f"Duration: {report.duration_s}s"
    )
    lines.append("")

    # Missing data
    if report.missing_data_issues:
        lines.append("Missing / Bad Data:")
        for issue in report.missing_data_issues:
            lines.append(f"  WARNING: {issue}")
        lines.append("")

    # Channel quality table
    lines.append("Channel Quality:")
    for cr in report.channel_results:
        # Determine RMS label (µV for raw data; normalised units otherwise)
        rms_label = f"RMS={cr.rms_uv:.1f}"
        status_str = f"{cr.status:<7}"
        snr_str = f"SNR={cr.snr_db:.1f}dB"
        clip_str = f"Clipping={cr.clipping_pct:.1f}%"

        base_line = f"  {cr.channel}: {status_str} {rms_label}  {snr_str}  {clip_str}"
        if cr.warnings:
            base_line += f"  [{cr.warnings[0]}]"
        lines.append(base_line)
        # Extra warnings on subsequent lines
        for w in cr.warnings[1:]:
            lines.append(f"           [{w}]")

    lines.append("")

    # Label distribution
    lines.append("Label Distribution:")
    if report.label_window_counts:
        # Format in rows of 6 for compactness
        items = sorted(report.label_window_counts.items())
        row_width = 6
        for i in range(0, len(items), row_width):
            row = items[i: i + row_width]
            row_str = "  " + "  ".join(f"{lbl}: {cnt} windows" for lbl, cnt in row)
            lines.append(row_str)

        # Warn about low-count classes
        for lbl, cnt in items:
            if cnt < _DEFAULT_MIN_WINDOWS:
                lines.append(
                    f"  WARNING: {lbl} only has {cnt} window(s) "
                    f"(minimum: {_DEFAULT_MIN_WINDOWS})"
                )
    else:
        lines.append("  (No complete windows found — check label coverage)")

    lines.append("")
    lines.append(f"Overall: {report.overall_status}")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def fix_session(filepath: str | Path) -> Path:
    """
    Auto-fix minor data quality issues in a session CSV and save the result.

    Fixes applied
    -------------
    1. Remove rows containing NaN in any channel or label column.
    2. Remove rows containing inf values in any channel column.
    3. Clip extreme channel values to ±3 standard deviations (per channel).

    The fixed file is written to ``{original_stem}_fixed.csv`` in the same
    directory as the original.

    Parameters
    ----------
    filepath:
        Path to the session CSV to fix.

    Returns
    -------
    Path
        Path to the saved fixed CSV file.
    """
    path = Path(filepath)
    df = pd.read_csv(path)

    channel_cols = [f"ch{i + 1}" for i in range(N_CHANNELS)]
    present_ch_cols = [c for c in channel_cols if c in df.columns]

    n_before = len(df)
    fixes_applied: List[str] = []

    # 1. Remove NaN rows
    df_clean = df.dropna(subset=present_ch_cols + (["label"] if "label" in df.columns else []))
    n_nan_removed = n_before - len(df_clean)
    if n_nan_removed > 0:
        fixes_applied.append(f"Removed {n_nan_removed} NaN rows")
    df = df_clean.reset_index(drop=True)

    # 2. Remove inf rows
    n_before_inf = len(df)
    inf_mask = np.zeros(len(df), dtype=bool)
    for col in present_ch_cols:
        inf_mask |= ~np.isfinite(df[col].values.astype(float))
    df = df[~inf_mask].reset_index(drop=True)
    n_inf_removed = n_before_inf - len(df)
    if n_inf_removed > 0:
        fixes_applied.append(f"Removed {n_inf_removed} inf rows")

    # 3. Clip extreme values to ±3 std dev per channel
    n_clipped_total = 0
    for col in present_ch_cols:
        vals = df[col].values.astype(np.float64)
        mean, std = vals.mean(), vals.std()
        if std > 0:
            low, high = mean - 3 * std, mean + 3 * std
            n_clipped = int(np.sum((vals < low) | (vals > high)))
            if n_clipped > 0:
                df[col] = np.clip(vals, low, high)
                n_clipped_total += n_clipped
    if n_clipped_total > 0:
        fixes_applied.append(f"Clipped {n_clipped_total} extreme sample(s) to ±3σ")

    # Save fixed file
    out_path = path.parent / f"{path.stem}_fixed{path.suffix}"
    df.to_csv(out_path, index=False)

    print(f"[fix_session] {path.name} → {out_path.name}")
    if fixes_applied:
        for fix in fixes_applied:
            print(f"  Applied: {fix}")
    else:
        print("  No fixes needed — file was already clean.")
    print(f"  Rows: {n_before} → {len(df)}")

    return out_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_session.py",
        description=(
            "Validate EMG session CSV files for signal quality issues. "
            "Accepts a single file or a directory of CSV files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Validate a single session file
  python scripts/validate_session.py data/raw/P001_20260227_000000.csv

  # Validate an entire directory
  python scripts/validate_session.py data/raw/

  # Auto-fix and save corrected files
  python scripts/validate_session.py data/raw/P001_20260227_000000.csv --fix

  # Stricter class balance requirement
  python scripts/validate_session.py data/raw/ --min-windows 10

  # Lower clipping threshold
  python scripts/validate_session.py data/raw/ --clip-threshold 0.5
        """,
    )

    parser.add_argument(
        "path",
        metavar="PATH",
        type=Path,
        help=(
            "Path to a single session CSV file, or a directory "
            "of CSV files to validate."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        default=False,
        help=(
            "Auto-fix minor issues: remove NaN/inf rows, clip extreme values "
            "to ±3σ per channel.  Saves fixed file as {name}_fixed.csv."
        ),
    )
    parser.add_argument(
        "--clip-threshold",
        metavar="PCT",
        type=float,
        default=_DEFAULT_CLIP_THRESHOLD_PCT,
        help=(
            "Percentage of samples allowed at ADC min/max before a clipping "
            f"warning is raised.  [default: {_DEFAULT_CLIP_THRESHOLD_PCT}]"
        ),
    )
    parser.add_argument(
        "--min-windows",
        metavar="N",
        type=int,
        default=_DEFAULT_MIN_WINDOWS,
        help=(
            "Minimum number of complete windows required per label class. "
            f"[default: {_DEFAULT_MIN_WINDOWS}]"
        ),
    )
    parser.add_argument(
        "--powerline-ratio",
        metavar="RATIO",
        type=float,
        default=_DEFAULT_POWERLINE_RATIO,
        help=(
            "Ratio of 60 Hz spectral power to background power above which "
            f"a power-line noise warning is issued.  [default: {_DEFAULT_POWERLINE_RATIO}]"
        ),
    )
    parser.add_argument(
        "--rms-low",
        metavar="UV",
        type=float,
        default=_DEFAULT_RMS_LOW_UV,
        help=(
            "RMS power (µV) below which a low-signal warning is issued "
            f"(raw data only).  [default: {_DEFAULT_RMS_LOW_UV}]"
        ),
    )
    parser.add_argument(
        "--rms-high",
        metavar="UV",
        type=float,
        default=_DEFAULT_RMS_HIGH_UV,
        help=(
            "RMS power (µV) above which a saturation warning is issued "
            f"(raw data only).  [default: {_DEFAULT_RMS_HIGH_UV}]"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the validation pipeline.

    Returns
    -------
    int
        Exit code: 0 if all sessions pass (with or without warnings),
        1 if any session fails with errors or cannot be read.
    """
    parser = _build_parser()
    args = parser.parse_args()

    target: Path = args.path

    if not target.exists():
        print(f"ERROR: Path does not exist: {target}", file=sys.stderr)
        return 1

    # Collect files to validate
    if target.is_dir():
        csv_files = sorted(target.glob("*.csv"))
        # Exclude *_fixed.csv from re-validation (auto-generated by --fix)
        csv_files = [f for f in csv_files if not f.stem.endswith("_fixed")]
        if not csv_files:
            print(f"ERROR: No CSV files found in '{target}'.", file=sys.stderr)
            return 1
    elif target.suffix.lower() == ".csv":
        csv_files = [target]
    else:
        print(
            f"ERROR: '{target}' is not a CSV file or directory.",
            file=sys.stderr,
        )
        return 1

    overall_exit_code = 0

    for csv_path in csv_files:
        # Validate
        try:
            report = validate_session(
                csv_path,
                clip_threshold_pct=args.clip_threshold,
                rms_low_uv=args.rms_low,
                rms_high_uv=args.rms_high,
                powerline_ratio=args.powerline_ratio,
                min_windows=args.min_windows,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"\nERROR validating '{csv_path.name}': {exc}", file=sys.stderr)
            overall_exit_code = 1
            continue

        print(f"\n{format_report(report)}")

        if report.n_errors > 0:
            overall_exit_code = 1

        # Auto-fix if requested
        if args.fix:
            print()
            try:
                fix_session(csv_path)
            except Exception as exc:
                print(f"  [fix_session] ERROR: {exc}", file=sys.stderr)

    return overall_exit_code


if __name__ == "__main__":
    sys.exit(main())

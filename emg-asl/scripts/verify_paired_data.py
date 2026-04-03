#!/usr/bin/env python3
"""
Verify temporal alignment and coverage of a paired EMG + video session.

Given an EMG CSV produced by record_paired_session.py, this script
automatically locates the corresponding _video.mp4 and _labels.csv files,
then checks:
  1. Temporal alignment  -- plots EMG ch1 vs vision labels on the same axis.
  2. Coverage            -- what % of EMG rows have a vision label.
  3. Latency             -- typical gap between an EMG timestamp and the
                            nearest video-frame timestamp.

Saves a diagnostic plot to docs/paired_data_verification.png (or the path
specified with --output).

Usage:
  python scripts/verify_paired_data.py data/raw/P001_20260301_143022.csv

  # Explicit overrides:
  python scripts/verify_paired_data.py data/raw/P001_20260301_143022.csv \\
      --video  data/raw/P001_20260301_143022_video.mp4 \\
      --labels data/raw/P001_20260301_143022_labels.csv \\
      --output docs/my_verification.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_companion(emg_path: Path, suffix: str) -> Path:
    """
    Given data/raw/P001_20260301_143022.csv, return
    data/raw/P001_20260301_143022{suffix}.
    """
    return emg_path.parent / (emg_path.stem + suffix)


# ---------------------------------------------------------------------------
# Video frame timestamp extraction
# ---------------------------------------------------------------------------

def _extract_video_frame_timestamps(video_path: Path) -> "np.ndarray":
    """
    Open the video file and collect the CAP_PROP_POS_MSEC timestamp of each
    frame. Returns a 1-D float64 array of millisecond timestamps.
    """
    if not _CV2_AVAILABLE:
        return np.array([], dtype=np.float64)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open video file: {video_path}")
        return np.array([], dtype=np.float64)

    timestamps = []
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()
    return np.array(timestamps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def _coverage_stats(emg_df: "pd.DataFrame") -> dict:
    """
    Compute label coverage from the 'label' column if present, or from
    vision_confidence (non-NaN rows).
    """
    n_total = len(emg_df)
    if n_total == 0:
        return {"total": 0, "labeled": 0, "coverage_pct": 0.0}

    if "vision_confidence" in emg_df.columns:
        n_labeled = int(emg_df["vision_confidence"].notna().sum())
    elif "label" in emg_df.columns:
        n_labeled = int((emg_df["label"] != "UNLABELED").sum())
    else:
        n_labeled = 0

    return {
        "total": n_total,
        "labeled": n_labeled,
        "coverage_pct": 100.0 * n_labeled / n_total,
    }


def _latency_stats(
    emg_ts: "np.ndarray",
    frame_ts: "np.ndarray",
) -> dict:
    """
    For each EMG timestamp, find the nearest video frame timestamp and compute
    the absolute gap. Returns summary statistics.
    """
    if len(frame_ts) == 0 or len(emg_ts) == 0:
        return {
            "n_pairs": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p95_ms": float("nan"),
            "max_ms": float("nan"),
        }

    sorted_frame_ts = np.sort(frame_ts)
    indices = np.searchsorted(sorted_frame_ts, emg_ts, side="left")
    # Clip to valid range
    indices = np.clip(indices, 0, len(sorted_frame_ts) - 1)
    # Also check left neighbor
    left_indices = np.clip(indices - 1, 0, len(sorted_frame_ts) - 1)

    gap_right = np.abs(sorted_frame_ts[indices] - emg_ts)
    gap_left = np.abs(sorted_frame_ts[left_indices] - emg_ts)
    gaps = np.minimum(gap_right, gap_left)

    return {
        "n_pairs": len(gaps),
        "mean_ms": float(np.mean(gaps)),
        "median_ms": float(np.median(gaps)),
        "p95_ms": float(np.percentile(gaps, 95)),
        "max_ms": float(np.max(gaps)),
    }


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _build_label_color_map(labels: "pd.Series") -> dict:
    """Assign a distinct color to each unique label string."""
    unique = sorted(labels.dropna().unique())
    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
    return {lbl: cmap(i) for i, lbl in enumerate(unique)}


def _generate_plot(
    emg_df: "pd.DataFrame",
    labels_df: "pd.DataFrame",
    frame_ts: "np.ndarray",
    latency: dict,
    coverage: dict,
    output_path: Path,
) -> None:
    """
    Build and save the diagnostic figure:

    Row 1: EMG ch1 over time (full session)
    Row 2: Vision label timeline (colored spans)
    Row 3: Per-EMG-row latency to nearest video frame
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"Paired Data Verification\n"
        f"Coverage: {coverage['labeled']}/{coverage['total']} EMG rows "
        f"({coverage['coverage_pct']:.1f}%)  |  "
        f"Latency median: {latency['median_ms']:.1f} ms  "
        f"p95: {latency['p95_ms']:.1f} ms",
        fontsize=12,
    )

    emg_ts = emg_df["timestamp_ms"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Panel 1: EMG ch1 raw signal
    # ------------------------------------------------------------------
    ax1 = axes[0]
    if "ch1" in emg_df.columns:
        ax1.plot(emg_ts / 1000.0, emg_df["ch1"].to_numpy(), lw=0.6, color="#2196F3", alpha=0.8)
    ax1.set_ylabel("ch1 amplitude (mV)")
    ax1.set_title("EMG channel 1 -- raw signal")
    ax1.set_xlabel("time (s)")

    # Shade labeled windows
    if "label" in emg_df.columns and "vision_confidence" in emg_df.columns:
        color_map = _build_label_color_map(emg_df["label"])
        prev_label = None
        span_start = None
        for i, (ts, lbl) in enumerate(zip(emg_ts, emg_df["label"])):
            if lbl == "UNLABELED" or pd.isna(lbl):
                if prev_label is not None:
                    ax1.axvspan(
                        span_start / 1000.0,
                        emg_ts[i - 1] / 1000.0,
                        alpha=0.15,
                        color=color_map.get(prev_label, "gray"),
                    )
                    prev_label = None
                    span_start = None
            else:
                if prev_label != lbl:
                    if prev_label is not None:
                        ax1.axvspan(
                            span_start / 1000.0,
                            ts / 1000.0,
                            alpha=0.15,
                            color=color_map.get(prev_label, "gray"),
                        )
                    prev_label = lbl
                    span_start = ts

        if prev_label is not None:
            ax1.axvspan(
                span_start / 1000.0,
                emg_ts[-1] / 1000.0,
                alpha=0.15,
                color=color_map.get(prev_label, "gray"),
            )

    # ------------------------------------------------------------------
    # Panel 2: Vision label timeline
    # ------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_title("Vision label timeline (colored by label)")
    ax2.set_ylabel("label")
    ax2.set_xlabel("time (s)")

    if not labels_df.empty and "timestamp_ms" in labels_df.columns:
        vis_ts = labels_df["timestamp_ms"].to_numpy(dtype=np.float64)
        vis_labels = labels_df["label"].to_numpy()
        vis_conf = labels_df.get("vision_confidence", labels_df.get("confidence", None))

        color_map2 = _build_label_color_map(labels_df["label"])
        label_list = sorted(labels_df["label"].dropna().unique())
        label_to_int = {lbl: i for i, lbl in enumerate(label_list)}

        y_vals = np.array([label_to_int.get(l, -1) for l in vis_labels])
        colors2 = [color_map2.get(l, "gray") for l in vis_labels]

        ax2.scatter(vis_ts / 1000.0, y_vals, c=colors2, s=4, alpha=0.7)
        ax2.set_yticks(list(range(len(label_list))))
        ax2.set_yticklabels(label_list, fontsize=7)

        # Legend patches
        patches = [
            mpatches.Patch(color=color_map2[lbl], label=lbl)
            for lbl in label_list
        ]
        if patches:
            ax2.legend(
                handles=patches,
                loc="upper right",
                ncol=min(6, len(patches)),
                fontsize=6,
            )

        # Confidence as alpha-scaled scatter on a twin axis
        if vis_conf is not None:
            ax2b = ax2.twinx()
            ax2b.plot(
                vis_ts / 1000.0,
                vis_conf.to_numpy() if hasattr(vis_conf, "to_numpy") else vis_conf,
                lw=0.5,
                color="black",
                alpha=0.3,
            )
            ax2b.set_ylabel("confidence", fontsize=8)
            ax2b.set_ylim(0.0, 1.2)
    else:
        ax2.text(
            0.5, 0.5,
            "No vision labels file found.",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )

    # ------------------------------------------------------------------
    # Panel 3: Per-EMG-row latency to nearest video frame
    # ------------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_title(
        f"Latency to nearest video frame  "
        f"(median={latency['median_ms']:.1f} ms, p95={latency['p95_ms']:.1f} ms)"
    )
    ax3.set_ylabel("gap (ms)")
    ax3.set_xlabel("time (s)")

    if len(frame_ts) > 0 and len(emg_ts) > 0:
        sorted_frame_ts = np.sort(frame_ts)
        indices = np.searchsorted(sorted_frame_ts, emg_ts, side="left")
        indices = np.clip(indices, 0, len(sorted_frame_ts) - 1)
        left_indices = np.clip(indices - 1, 0, len(sorted_frame_ts) - 1)
        gap_right = np.abs(sorted_frame_ts[indices] - emg_ts)
        gap_left = np.abs(sorted_frame_ts[left_indices] - emg_ts)
        gaps = np.minimum(gap_right, gap_left)

        # Subsample for visual clarity if session is large
        step = max(1, len(emg_ts) // 2000)
        ax3.plot(
            emg_ts[::step] / 1000.0,
            gaps[::step],
            lw=0.6,
            color="#E91E63",
            alpha=0.7,
        )
        ax3.axhline(latency["median_ms"], color="orange", lw=1.2, linestyle="--", label="median")
        ax3.axhline(latency["p95_ms"], color="red", lw=1.0, linestyle=":", label="p95")
        ax3.legend(fontsize=8)
        ax3.set_ylim(bottom=0)
    else:
        ax3.text(
            0.5, 0.5,
            "No video frame timestamps available.",
            transform=ax3.transAxes,
            ha="center",
            va="center",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostic plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main verification driver
# ---------------------------------------------------------------------------

def verify(
    emg_csv: Path,
    video_mp4: Optional[Path] = None,
    labels_csv: Optional[Path] = None,
    output_png: Optional[Path] = None,
) -> None:
    """
    Run the full verification suite for a paired session.

    Parameters
    ----------
    emg_csv:
        Path to the primary EMG CSV produced by record_paired_session.py.
    video_mp4:
        Path to the companion video file. Auto-detected from emg_csv stem if
        None.
    labels_csv:
        Path to the companion labels CSV. Auto-detected if None.
    output_png:
        Destination for the diagnostic plot. Defaults to
        docs/paired_data_verification.png relative to the project root.
    """
    if not _PD_AVAILABLE or not _NP_AVAILABLE:
        print("[ERROR] pandas and numpy are required: pip install pandas numpy")
        sys.exit(1)

    if not _MPL_AVAILABLE:
        print("[ERROR] matplotlib is required: pip install matplotlib")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Locate companion files
    # ------------------------------------------------------------------
    if video_mp4 is None:
        video_mp4 = _find_companion(emg_csv, "_video.mp4")
    if labels_csv is None:
        labels_csv = _find_companion(emg_csv, "_labels.csv")
    if output_png is None:
        output_png = _PROJECT_ROOT / "docs" / "paired_data_verification.png"

    print("=" * 60)
    print("Paired Data Verification")
    print("=" * 60)
    print(f"  EMG CSV    : {emg_csv}")
    print(f"  Video MP4  : {video_mp4}  {'[FOUND]' if video_mp4.exists() else '[NOT FOUND]'}")
    print(f"  Labels CSV : {labels_csv}  {'[FOUND]' if labels_csv.exists() else '[NOT FOUND]'}")
    print(f"  Plot output: {output_png}")
    print()

    # ------------------------------------------------------------------
    # Load EMG CSV
    # ------------------------------------------------------------------
    if not emg_csv.exists():
        print(f"[ERROR] EMG CSV not found: {emg_csv}")
        sys.exit(1)

    print("Loading EMG CSV...")
    emg_df = pd.read_csv(emg_csv)
    print(f"  {len(emg_df)} rows, columns: {list(emg_df.columns)}")

    if "timestamp_ms" not in emg_df.columns:
        print("[ERROR] EMG CSV is missing 'timestamp_ms' column.")
        sys.exit(1)

    emg_ts = emg_df["timestamp_ms"].to_numpy(dtype=np.float64)
    session_duration_s = (emg_ts[-1] - emg_ts[0]) / 1000.0 if len(emg_ts) > 1 else 0.0
    print(f"  Session duration: {session_duration_s:.2f} s")
    print(f"  Timestamp range : {emg_ts[0]:.1f} ms  to  {emg_ts[-1]:.1f} ms")

    # ------------------------------------------------------------------
    # Load labels CSV
    # ------------------------------------------------------------------
    print("\nLoading vision labels CSV...")
    if labels_csv.exists():
        labels_df = pd.read_csv(labels_csv)
        print(f"  {len(labels_df)} rows, columns: {list(labels_df.columns)}")
        if not labels_df.empty:
            unique_labels = sorted(labels_df["label"].dropna().unique())
            print(f"  Unique labels: {unique_labels}")
    else:
        labels_df = pd.DataFrame(columns=["timestamp_ms", "label", "vision_confidence"])
        print("  [WARN] Labels CSV not found. Coverage and alignment checks will be limited.")

    # ------------------------------------------------------------------
    # Extract video frame timestamps
    # ------------------------------------------------------------------
    print("\nExtracting video frame timestamps...")
    if video_mp4.exists() and _CV2_AVAILABLE:
        frame_ts = _extract_video_frame_timestamps(video_mp4)
        print(f"  {len(frame_ts)} frames  "
              f"(duration {frame_ts[-1]/1000.0:.2f} s)" if len(frame_ts) > 0 else "  0 frames")
    else:
        frame_ts = np.array([], dtype=np.float64)
        if not video_mp4.exists():
            print("  [WARN] Video file not found. Latency check will be skipped.")
        elif not _CV2_AVAILABLE:
            print("  [WARN] opencv-python not installed. Skipping frame timestamp extraction.")

    # ------------------------------------------------------------------
    # Coverage check
    # ------------------------------------------------------------------
    print("\nCoverage check...")
    # Merge vision labels into EMG if available
    if not labels_df.empty:
        from src.data.vision_teacher import sync_labels_to_emg
        merged = sync_labels_to_emg(labels_df, emg_df, tolerance_ms=50.0)
    else:
        merged = emg_df.copy()
        if "label" not in merged.columns:
            merged["label"] = "UNLABELED"
        if "vision_confidence" not in merged.columns:
            merged["vision_confidence"] = float("nan")

    coverage = _coverage_stats(merged)
    print(f"  Labeled EMG rows : {coverage['labeled']}/{coverage['total']} "
          f"({coverage['coverage_pct']:.1f}%)")

    # Per-label breakdown
    if "label" in merged.columns:
        from collections import Counter
        label_counts = Counter(
            lbl for lbl in merged["label"] if lbl != "UNLABELED"
        )
        if label_counts:
            print("  Per-label breakdown:")
            for lbl, cnt in sorted(label_counts.items()):
                print(f"    {lbl:>12s} : {cnt:5d} labeled EMG rows")

    # ------------------------------------------------------------------
    # Latency check
    # ------------------------------------------------------------------
    print("\nLatency check (EMG timestamp vs nearest video frame)...")
    latency = _latency_stats(emg_ts, frame_ts)
    if latency["n_pairs"] > 0:
        print(f"  Mean   : {latency['mean_ms']:.2f} ms")
        print(f"  Median : {latency['median_ms']:.2f} ms")
        print(f"  p95    : {latency['p95_ms']:.2f} ms")
        print(f"  Max    : {latency['max_ms']:.2f} ms")

        # Sanity threshold: at 30 fps the frame interval is ~33 ms.
        # If median latency exceeds 100 ms there may be a clock-drift problem.
        if latency["median_ms"] > 100.0:
            print("  [WARN] Median latency > 100 ms -- check wall-clock sync between threads.")
        else:
            print("  [OK] Latency within acceptable range.")
    else:
        print("  [INFO] No frame timestamps available for latency analysis.")

    # ------------------------------------------------------------------
    # Temporal alignment check: compare EMG and video duration
    # ------------------------------------------------------------------
    print("\nTemporal alignment check...")
    if len(frame_ts) > 0:
        video_duration_s = (frame_ts[-1] - frame_ts[0]) / 1000.0
        drift_s = abs(session_duration_s - video_duration_s)
        print(f"  EMG duration   : {session_duration_s:.3f} s")
        print(f"  Video duration : {video_duration_s:.3f} s")
        print(f"  Duration drift : {drift_s * 1000.0:.1f} ms")
        if drift_s > 0.5:
            print("  [WARN] Duration drift > 500 ms -- streams may not be in lockstep.")
        else:
            print("  [OK] Durations match within 500 ms.")
    else:
        print("  [INFO] No video data available for duration comparison.")

    # ------------------------------------------------------------------
    # Generate diagnostic plot
    # ------------------------------------------------------------------
    print("\nGenerating diagnostic plot...")
    _generate_plot(
        emg_df=merged,
        labels_df=labels_df,
        frame_ts=frame_ts,
        latency=latency,
        coverage=coverage,
        output_path=output_png,
    )

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"  Coverage  : {coverage['coverage_pct']:.1f}%")
    if latency["n_pairs"] > 0:
        print(f"  Latency   : median {latency['median_ms']:.1f} ms, p95 {latency['p95_ms']:.1f} ms")
    print(f"  Plot      : {output_png}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify temporal alignment of a paired EMG + video session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_paired_data.py data/raw/P001_20260301_143022.csv

  python scripts/verify_paired_data.py data/raw/P001_20260301_143022.csv \\
      --video  data/raw/P001_20260301_143022_video.mp4 \\
      --labels data/raw/P001_20260301_143022_labels.csv \\
      --output docs/my_verification.png
        """,
    )

    parser.add_argument(
        "emg_csv",
        type=Path,
        help="Path to the EMG session CSV (e.g. data/raw/P001_20260301_143022.csv).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        dest="video",
        help="Companion video file. Auto-detected from EMG CSV stem if not given.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        dest="labels",
        help="Companion labels CSV. Auto-detected from EMG CSV stem if not given.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        dest="output",
        help="Output PNG path. Defaults to docs/paired_data_verification.png.",
    )

    args = parser.parse_args()
    verify(
        emg_csv=args.emg_csv,
        video_mp4=args.video,
        labels_csv=args.labels,
        output_png=args.output,
    )


if __name__ == "__main__":
    main()

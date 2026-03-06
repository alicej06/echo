"""
Record a labeled EMG session from the Thalmic MYO Armband via myo-python.

Requirements
------------
- MyoConnect must be running (connects to MYO via USB dongle).
- myo-python must be installed: pip install myo-python
- MYO_SDK_PATH env var must point to the Myo C++ SDK directory, or pass --sdk-path.

See hardware/myo_armband/README.md for full setup instructions.

Usage (from project root)
--------------------------
    python scripts/record_session.py --participant P001 --labels A B C D E --reps 5
    python scripts/record_session.py --participant P001 --all-letters --reps 10
    python scripts/record_session.py --participant P001 --all-labels --reps 5

Output CSV columns
------------------
timestamp_ms, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, label
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.*`` imports work when
# invoked as ``python scripts/record_session.py`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import myo
    _MYO_AVAILABLE = True
except ImportError:
    _MYO_AVAILABLE = False

from src.utils.constants import (
    ASL_LABELS,
    ASL_LETTERS,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# myo-python listener
# ---------------------------------------------------------------------------


if _MYO_AVAILABLE:
    class _RecordingListener(myo.DeviceListener):
        """Collects labeled EMG samples from the MYO armband."""

        def __init__(self) -> None:
            self.samples: list = []          # List of (ts_ms, [ch0..ch7], label)
            self._collecting = False
            self._current_label = ""
            self._session_start_ms = time.monotonic() * 1000.0

        def start_collecting(self, label: str) -> None:
            self._current_label = label
            self._collecting = True

        def stop_collecting(self) -> None:
            self._collecting = False

        def on_emg_data(self, myo_device, timestamp, emg_data) -> None:  # type: ignore[override]
            if not self._collecting:
                return
            ts_ms = time.monotonic() * 1000.0 - self._session_start_ms
            # Scale int8 (-128..127) to mV using MYO full-scale ~±1.25 mV
            mv = [v / 127.0 * 1250.0 for v in emg_data]
            self.samples.append((ts_ms, mv, self._current_label))

        def on_connected(self, myo_device, timestamp, firmware_version) -> None:  # type: ignore[override]
            myo_device.set_stream_emg(myo.StreamEmg.enabled)
            print(f"[MYO] Connected (firmware {firmware_version}) — EMG streaming enabled")

        def on_disconnected(self, myo_device, timestamp) -> None:  # type: ignore[override]
            print("[MYO] Disconnected.")


# ---------------------------------------------------------------------------
# Session recording helpers
# ---------------------------------------------------------------------------


def _build_label_list(args: argparse.Namespace) -> list:
    """Resolve the label list from CLI arguments."""
    if args.all_labels:
        return list(ASL_LABELS)
    if args.all_letters:
        return list(ASL_LETTERS)
    if args.labels:
        invalid = [lbl for lbl in args.labels if lbl not in ASL_LABELS]
        if invalid:
            raise ValueError(f"Unknown label(s): {invalid}\nValid labels: {ASL_LABELS}")
        return args.labels
    raise ValueError("Specify at least one of --labels, --all-letters, or --all-labels.")


def _save_csv(samples: list, participant_id: str, output_dir: str) -> Path:
    """Save collected samples to a timestamped CSV file."""
    import csv
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = time.strftime("%Y%m%d_%H%M%S")
    filepath = out_dir / f"{participant_id}_{date_str}.csv"
    n_channels = 8
    headers = ["timestamp_ms"] + [f"ch{i + 1}" for i in range(n_channels)] + ["label"]
    with filepath.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for ts_ms, mv, label in samples:
            row: dict = {"timestamp_ms": f"{ts_ms:.3f}", "label": label}
            for i, val in enumerate(mv):
                row[f"ch{i + 1}"] = f"{val:.6f}"
            writer.writerow(row)
    return filepath.resolve()


def _run(args: argparse.Namespace) -> None:
    if not _MYO_AVAILABLE:
        print(
            "ERROR: myo-python is not installed.\n"
            "  Run: pip install myo-python\n"
            "  Also ensure MyoConnect is running and MYO_SDK_PATH is set.\n"
            "  See hardware/myo_armband/README.md"
        )
        sys.exit(1)

    labels = _build_label_list(args)

    print("=" * 60)
    print("EMG-ASL Recording Session — MYO Armband")
    print("=" * 60)
    print(f"Participant   : {args.participant}")
    print(f"Output dir    : {args.output}")
    print(f"Labels        : {labels}")
    print(f"Reps per label: {args.reps}")
    print(f"Hold duration : {args.hold_duration} s")
    print(f"SDK path      : {args.sdk_path or '(from MYO_SDK_PATH env)'}")
    print()
    print("[INFO] Ensure MyoConnect is running and the MYO armband is charged.")
    print("[INFO] Put on the armband — logo facing outward, 2-3 cm below elbow.")
    print()

    # Initialise MYO SDK
    if args.sdk_path:
        myo.init(sdk_path=args.sdk_path)
    else:
        myo.init()

    listener = _RecordingListener()
    hub = myo.Hub()
    hub.run_in_background(listener.on_event)

    print("[INFO] Waiting for MYO connection (LED will pulse blue)...")
    time.sleep(3.0)  # Give the SDK time to connect

    try:
        for label in labels:
            print(f"\n{'=' * 40}")
            print(f"LABEL: {label}  ({labels.index(label) + 1}/{len(labels)})")
            print(f"{'=' * 40}")

            for rep in range(1, args.reps + 1):
                print(f"\nGET READY: [{label}]  (rep {rep}/{args.reps})  - press ENTER when ready")
                input()

                # 3-second countdown
                for i in range(3, 0, -1):
                    print(f"  {i}…", end=" ", flush=True)
                    time.sleep(1.0)
                print("GO!", flush=True)

                print(f"  Recording '{label}' rep {rep}/{args.reps} …", end=" ", flush=True)
                listener.start_collecting(label)
                time.sleep(args.hold_duration)
                listener.stop_collecting()

                n_this_rep = sum(1 for _, _, lbl in listener.samples if lbl == label)
                print(f"done  (~{SAMPLE_RATE} Hz target, {int(args.hold_duration * SAMPLE_RATE)} expected)")

                if rep < args.reps:
                    print(f"  Rest 1.5 s …", end=" ", flush=True)
                    time.sleep(1.5)
                    print("ready.")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving partial data…")
    finally:
        hub.shutdown()

    if not listener.samples:
        print("\n[ERROR] No samples recorded. Check MYO connection and try again.")
        sys.exit(1)

    filepath = _save_csv(listener.samples, args.participant, args.output)

    # Summary
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    label_counts = Counter(lbl for _, _, lbl in listener.samples)
    for lbl in labels:
        count = label_counts.get(lbl, 0)
        expected = args.reps * int(args.hold_duration * SAMPLE_RATE)
        print(f"  {lbl:>12s} : {count:5d} samples  (expected ~{expected})")
    total = sum(label_counts.values())
    print(f"\n  Total samples : {total}")
    print(f"  Saved to      : {filepath}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a labeled EMG session from the Thalmic MYO Armband.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/record_session.py --participant P001 --labels A B C --reps 5
  python scripts/record_session.py --participant P001 --all-letters --reps 10
  python scripts/record_session.py --participant P001 --all-labels --reps 5
        """,
    )
    parser.add_argument("--participant", required=True,
                        help="Participant ID (e.g. P001). Used in output filename.")
    parser.add_argument("--output", default="data/raw/",
                        help="Output directory for the session CSV (default: data/raw/).")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="One or more ASL label strings to record (e.g. A B C HELLO).")
    parser.add_argument("--all-letters", action="store_true",
                        help="Record all 26 ASL letters (A-Z).")
    parser.add_argument("--all-labels", action="store_true",
                        help="Record all 36 ASL classes (26 letters + 10 words).")
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of repetitions per label (default: 10).")
    parser.add_argument("--hold-duration", type=float, default=2.0, dest="hold_duration",
                        help="Recording duration per rep in seconds (default: 2.0).")
    parser.add_argument("--sdk-path", default="", dest="sdk_path",
                        help="Path to the Myo C++ SDK directory (overrides MYO_SDK_PATH env).")

    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()

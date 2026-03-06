#!/usr/bin/env python3
"""
prepare_italian_sl.py — Italian Sign Language EMG dataset → EMG-ASL session CSV converter.

Usage
-----
    # Print download / clone instructions (no arguments required)
    python scripts/prepare_italian_sl.py

    # Convert after cloning the airtlab repository
    python scripts/prepare_italian_sl.py --data-dir /path/to/repo/data/

    # Optional: customise output directory
    python scripts/prepare_italian_sl.py \\
        --data-dir /path/to/repo/data/ \\
        --output-dir data/raw/italian_sl

The script always prints the git clone instructions first, then (if --data-dir
is given) runs the conversion.

Output CSVs (ISL001_italian_sl_YYYYMMDD.csv, ISL002, ISL003) are immediately
compatible with src/data/loader.py::load_dataset().

Dataset source
--------------
    git clone https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet

Pass the ``data/`` subdirectory of the cloned repository to --data-dir.
Expected layout inside that directory:  A/ B/ ... Z/ (each with 30 JSON files)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so that the script works when
# run directly with `python scripts/prepare_italian_sl.py` (i.e. without
# installing the package via `pip install -e .`).
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.italian_sl_adapter import (  # noqa: E402
    convert_italian_sl_dataset,
    download_italian_sl_instructions,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_italian_sl.py",
        description=(
            "Print Italian Sign Language (airtlab) dataset clone instructions "
            "and optionally convert the JSON files to EMG-ASL session CSV format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Show download / clone instructions only
  python scripts/prepare_italian_sl.py

  # Convert after cloning the repository
  python scripts/prepare_italian_sl.py \\
      --data-dir /path/to/An-EMG-and-IMU-Dataset.../data/

  # Override the default output directory
  python scripts/prepare_italian_sl.py \\
      --data-dir /path/to/repo/data/ \\
      --output-dir data/raw/italian_sl
        """,
    )

    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help=(
            "Path to the 'data/' subdirectory of the cloned airtlab repository. "
            "Must contain subdirectories A/, B/, ..., Z/ each with 30 JSON files. "
            "When omitted, only the clone instructions are printed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=Path("data/raw/italian_sl"),
        help=(
            "Directory where converted session CSV files will be written. "
            "Created automatically if it does not exist. "
            "[default: data/raw/italian_sl]"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the Italian Sign Language dataset preparation pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Always print the clone instructions so users know where the data
    # comes from and how to obtain it.
    print(download_italian_sl_instructions())

    if args.data_dir is None:
        print(
            "Tip: run with --data-dir to convert the JSON files after cloning.\n"
            "     python scripts/prepare_italian_sl.py \\\n"
            "         --data-dir /path/to/An-EMG-and-IMU-Dataset.../data/\n"
        )
        return 0

    # Validate the supplied directory before attempting conversion.
    if not args.data_dir.exists():
        print(
            f"ERROR: --data-dir does not exist: {args.data_dir}\n"
            "       Clone the repository and pass the 'data/' subdirectory "
            "to --data-dir.",
            file=sys.stderr,
        )
        return 1

    if not args.data_dir.is_dir():
        print(
            f"ERROR: --data-dir is not a directory: {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    # Run the conversion.
    try:
        convert_italian_sl_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"ERROR: Unexpected failure during conversion: {exc}",
            file=sys.stderr,
        )
        raise  # re-raise so the traceback is visible for debugging

    print(
        f"\nConversion complete.  Session CSVs written to: "
        f"{args.output_dir.resolve()}\n"
        f"Three participant files were produced:\n"
        f"  ISL001_italian_sl_*.csv  (reps 1-10 per letter)\n"
        f"  ISL002_italian_sl_*.csv  (reps 11-20 per letter)\n"
        f"  ISL003_italian_sl_*.csv  (reps 21-30 per letter)\n"
        f"\nYou can load them with:\n"
        f"    from src.data.loader import load_dataset\n"
        f"    df = load_dataset('{args.output_dir}')\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

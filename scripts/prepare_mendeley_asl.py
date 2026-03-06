#!/usr/bin/env python3
"""
prepare_mendeley_asl.py — Mendeley ASL Myo dataset → EMG-ASL session CSV converter.

Usage
-----
    # Print download instructions (no arguments required)
    python scripts/prepare_mendeley_asl.py

    # Convert downloaded dataset
    python scripts/prepare_mendeley_asl.py --data-dir /path/to/mendeley_asl/

    # Optional: customise output directory
    python scripts/prepare_mendeley_asl.py \\
        --data-dir /path/to/mendeley_asl/ \\
        --output-dir data/raw/mendeley_asl

The script always prints the download instructions first, then (if --data-dir
is given) runs the conversion.

Output CSVs (MND001_mendeley_asl_YYYYMMDD.csv, MND002, ...) are immediately
compatible with src/data/loader.py::load_dataset().

Dataset source
--------------
    https://data.mendeley.com/datasets/wgswcr8z24/2

Download the ZIP from the Mendeley Data page (free, no registration required)
and unzip it.  Pass the unzipped root directory to --data-dir.

Expected layout after unzipping:
    mendeley_asl/
        user_1/
            A.csv
            HELLO.csv
            ...
        user_2/
            ...

Only the 36 ASL classes in the MAIA vocabulary are retained.  All other
signs are automatically skipped during conversion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so the script runs correctly when
# invoked directly with `python scripts/prepare_mendeley_asl.py` without
# a package installation.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.mendeley_asl_adapter import (  # noqa: E402
    convert_mendeley_dataset,
    download_mendeley_instructions,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_mendeley_asl.py",
        description=(
            "Print Mendeley ASL Myo dataset download instructions and "
            "optionally convert the CSV files to EMG-ASL session format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Show download instructions only
  python scripts/prepare_mendeley_asl.py

  # Convert after downloading and unzipping the dataset
  python scripts/prepare_mendeley_asl.py --data-dir /path/to/mendeley_asl/

  # Override the default output directory
  python scripts/prepare_mendeley_asl.py \\
      --data-dir /path/to/mendeley_asl/ \\
      --output-dir data/raw/mendeley_asl
        """,
    )

    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help=(
            "Root directory of the unzipped Mendeley ASL download. "
            "Should contain user_1/, user_2/, ... subdirectories (or CSV files "
            "directly).  When omitted, only the download instructions are printed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=Path("data/raw/mendeley_asl"),
        help=(
            "Directory where converted session CSV files will be written. "
            "Created automatically if it does not exist. "
            "[default: data/raw/mendeley_asl]"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the Mendeley ASL dataset preparation pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Always print download instructions so users know where the data
    # comes from and how to obtain it.
    print(download_mendeley_instructions())

    if args.data_dir is None:
        print(
            "Tip: run with --data-dir to convert the downloaded dataset.\n"
            "     python scripts/prepare_mendeley_asl.py \\\n"
            "         --data-dir /path/to/mendeley_asl/\n"
        )
        return 0

    # Validate the supplied directory before attempting conversion.
    if not args.data_dir.exists():
        print(
            f"ERROR: --data-dir does not exist: {args.data_dir}\n"
            "       Download and unzip the Mendeley dataset and pass the "
            "root directory to --data-dir.",
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
        convert_mendeley_dataset(
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
        f"\nYou can load them with:\n"
        f"    from src.data.loader import load_dataset\n"
        f"    df = load_dataset('{args.output_dir}')\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

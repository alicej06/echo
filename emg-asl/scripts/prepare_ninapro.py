#!/usr/bin/env python3
"""
prepare_ninapro.py — NinaProDB DB1 → EMG-ASL session CSV converter.

Usage
-----
    # Print download instructions (no arguments required)
    python scripts/prepare_ninapro.py

    # Convert downloaded .mat files to session CSVs
    python scripts/prepare_ninapro.py --mat-dir /path/to/ninapro/db1

    # Optional: customise output directory and subject prefix
    python scripts/prepare_ninapro.py \\
        --mat-dir /path/to/ninapro/db1 \\
        --output-dir data/raw/ninapro \\
        --subject-prefix NP

The script always prints download instructions first, then (if --mat-dir is
given) runs the conversion.  Output CSVs are written to --output-dir and are
immediately compatible with src/data/loader.py::load_dataset().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the script is run directly
# (i.e. `python scripts/prepare_ninapro.py`).  If the package is installed
# (pip install -e .) this block is a no-op.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.ninapro_adapter import (  # noqa: E402
    convert_ninapro_db1_to_sessions,
    download_ninapro_db1_instructions,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_ninapro.py",
        description=(
            "Print NinaProDB DB1 download instructions and optionally convert "
            "downloaded .mat files to EMG-ASL session CSV format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Show download instructions only
  python scripts/prepare_ninapro.py

  # Convert after downloading DB1 Exercise 1 files
  python scripts/prepare_ninapro.py --mat-dir /path/to/ninapro/db1

  # Override output directory and participant prefix
  python scripts/prepare_ninapro.py \\
      --mat-dir /path/to/ninapro/db1 \\
      --output-dir data/raw/ninapro \\
      --subject-prefix NP
        """,
    )

    parser.add_argument(
        "--mat-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help=(
            "Directory containing S*_E1_A1.mat files downloaded from NinaProDB. "
            "When omitted, only the download instructions are printed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=Path("data/raw/ninapro"),
        help=(
            "Directory where converted session CSV files will be written. "
            "Created automatically if it does not exist. "
            "[default: data/raw/ninapro]"
        ),
    )
    parser.add_argument(
        "--subject-prefix",
        metavar="PREFIX",
        type=str,
        default="NP",
        help=(
            "String prepended to subject numbers in output filenames "
            "(e.g. 'NP' → 'NP001_ninapro_20260227.csv'). "
            "[default: NP]"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the NinaProDB preparation pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Always print download instructions so users know where the data comes
    # from and how to obtain it, regardless of whether --mat-dir was given.
    print(download_ninapro_db1_instructions())

    if args.mat_dir is None:
        # No conversion requested — instructions already printed above.
        print(
            "Tip: run with --mat-dir to convert downloaded .mat files.\n"
            "     python scripts/prepare_ninapro.py --mat-dir /path/to/ninapro/db1\n"
        )
        return 0

    # Validate that the supplied directory exists before attempting conversion.
    if not args.mat_dir.exists():
        print(
            f"ERROR: --mat-dir does not exist: {args.mat_dir}\n"
            "       Download the DB1 Exercise 1 .mat files and pass the "
            "directory path to --mat-dir.",
            file=sys.stderr,
        )
        return 1

    if not args.mat_dir.is_dir():
        print(
            f"ERROR: --mat-dir is not a directory: {args.mat_dir}",
            file=sys.stderr,
        )
        return 1

    # Run the conversion.
    try:
        convert_ninapro_db1_to_sessions(
            mat_dir=args.mat_dir,
            output_dir=args.output_dir,
            subject_prefix=args.subject_prefix,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: Unexpected failure during conversion: {exc}", file=sys.stderr)
        raise  # re-raise so the traceback is visible for debugging

    print(
        f"\nConversion complete.  Session CSVs written to: {args.output_dir.resolve()}\n"
        f"You can now load them with:\n"
        f"    from src.data.loader import load_dataset\n"
        f"    df = load_dataset('{args.output_dir}')\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

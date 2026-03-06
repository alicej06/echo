#!/usr/bin/env python3
"""
prepare_asla.py — ASLA dataset → EMG-ASL session CSV converter.

Usage
-----
    # Print access / download instructions (no arguments required)
    python scripts/prepare_asla.py

    # Convert downloaded ASLA files to session CSVs
    python scripts/prepare_asla.py --data-dir /path/to/asla/raw

    # Specify left-handed subjects whose channels should be mirrored
    python scripts/prepare_asla.py \\
        --data-dir /path/to/asla/raw \\
        --left-handed S003 S011

    # Customise output directory
    python scripts/prepare_asla.py \\
        --data-dir /path/to/asla/raw \\
        --output-dir data/raw/asla \\
        --left-handed S003 S011

The script always prints access instructions first, then (if --data-dir is
given) runs the conversion.  Output CSVs are written to --output-dir and are
immediately compatible with src/data/loader.py::load_dataset().

Dataset: RIT MABL Lab — ASL Alphabet (ASLA)
Access:  Sign EULA and email feseee@rit.edu (see instructions below)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly (python scripts/...)
# If the package is installed (pip install -e .) this is a no-op.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asla_adapter import (  # noqa: E402
    convert_asla_dataset,
    download_asla_instructions,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_asla.py",
        description=(
            "Print ASLA dataset access instructions and optionally convert "
            "downloaded data files to EMG-ASL session CSV format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Show dataset access instructions only
  python scripts/prepare_asla.py

  # Convert downloaded files to session CSVs
  python scripts/prepare_asla.py --data-dir /path/to/asla/raw

  # Mirror channels for known left-handed subjects
  python scripts/prepare_asla.py \\
      --data-dir /path/to/asla/raw \\
      --left-handed S003 S011

  # Full options
  python scripts/prepare_asla.py \\
      --data-dir /path/to/asla/raw \\
      --output-dir data/raw/asla \\
      --left-handed S003 S011
        """,
    )

    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help=(
            "Directory containing ASLA data files (CSV, .mat, or .npy). "
            "When omitted, only the access instructions are printed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=Path("data/raw/asla"),
        help=(
            "Directory where converted session CSV files will be written. "
            "Created automatically if it does not exist. "
            "[default: data/raw/asla]"
        ),
    )
    parser.add_argument(
        "--left-handed",
        metavar="SUBJECT_ID",
        nargs="+",
        default=None,
        help=(
            "Space-separated list of subject IDs (e.g. S003 S011) for whom "
            "channel order should be reversed before saving.  The Myo armband "
            "is worn on the dominant arm; left-handed subjects wear it on the "
            "left arm, which reverses the anatomical muscle-group ordering. "
            "Omit if no handedness information is available."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the ASLA preparation pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Always print access instructions so users know how to obtain the dataset.
    print(download_asla_instructions())

    if args.data_dir is None:
        print(
            "Tip: run with --data-dir to convert downloaded ASLA files.\n"
            "     python scripts/prepare_asla.py --data-dir /path/to/asla/raw\n"
        )
        return 0

    # Validate the supplied directory before attempting conversion.
    if not args.data_dir.exists():
        print(
            f"ERROR: --data-dir does not exist: {args.data_dir}\n"
            "       Download the ASLA dataset (see instructions above) and "
            "pass the directory path to --data-dir.",
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
        convert_asla_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            left_handed_subjects=args.left_handed,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: Unexpected failure during conversion: {exc}", file=sys.stderr)
        raise  # Re-raise so the full traceback is visible for debugging

    print(
        f"\nConversion complete.  Session CSVs written to: {args.output_dir.resolve()}\n"
        f"You can now load them with:\n"
        f"    from src.data.loader import load_dataset\n"
        f"    df = load_dataset('{args.output_dir}')\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

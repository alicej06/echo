#!/usr/bin/env python3
"""
prepare_grabmyo.py — GRABMyo → EMG-ASL session CSV converter.

Usage
-----
    # Print download instructions only (no arguments required)
    python scripts/prepare_grabmyo.py

    # Convert a local GRABMyo download to session CSVs
    python scripts/prepare_grabmyo.py --data-dir /path/to/grabmyo/

    # Limit to the first 5 subjects
    python scripts/prepare_grabmyo.py --data-dir /path/to/grabmyo/ --max-subjects 5

    # Override output directory
    python scripts/prepare_grabmyo.py \\
        --data-dir /path/to/grabmyo/ \\
        --output-dir data/raw/grabmyo \\
        --max-subjects 10

The script always prints download instructions first so users know how to
obtain the data before running the conversion.  If --data-dir is omitted,
only the instructions are shown.

Output CSVs are written to --output-dir (default: data/raw/grabmyo/) and are
immediately compatible with src/data/loader.py::load_dataset().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the script is run directly
# (i.e. `python scripts/prepare_grabmyo.py`).  If the package is installed
# (pip install -e .) this block is a no-op.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.grabmyo_adapter import (  # noqa: E402
    convert_grabmyo_dataset,
    download_grabmyo_instructions,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_grabmyo.py",
        description=(
            "Print GRABMyo download instructions and optionally convert a "
            "local GRABMyo download to EMG-ASL session CSV format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Show download instructions only
  python scripts/prepare_grabmyo.py

  # Convert after downloading GRABMyo
  python scripts/prepare_grabmyo.py --data-dir /path/to/grabmyo/

  # Convert only the first 5 subjects
  python scripts/prepare_grabmyo.py --data-dir /path/to/grabmyo/ --max-subjects 5

  # Override output directory
  python scripts/prepare_grabmyo.py \\
      --data-dir /path/to/grabmyo/ \\
      --output-dir data/raw/grabmyo \\
      --max-subjects 10
        """,
    )

    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help=(
            "Root directory of the GRABMyo download.  Should contain "
            "'participant_N' subdirectories (or 'subject_N').  "
            "When omitted, only the download instructions are printed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        default=Path("data/raw/grabmyo"),
        help=(
            "Directory where converted session CSV files will be written. "
            "Created automatically if it does not exist. "
            "[default: data/raw/grabmyo]"
        ),
    )
    parser.add_argument(
        "--max-subjects",
        metavar="N",
        type=int,
        default=10,
        help=(
            "Maximum number of subjects to convert.  GRABMyo has 43 subjects; "
            "use --max-subjects 43 to convert all of them. "
            "[default: 10]"
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Run the GRABMyo preparation pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # Always print download instructions so users know where the data comes
    # from and how to obtain it, regardless of whether --data-dir was given.
    print(download_grabmyo_instructions())

    if args.data_dir is None:
        print(
            "Tip: run with --data-dir to convert a local GRABMyo download.\n"
            "     python scripts/prepare_grabmyo.py --data-dir /path/to/grabmyo/\n"
        )
        return 0

    # Validate the supplied directory before attempting conversion.
    if not args.data_dir.exists():
        print(
            f"ERROR: --data-dir does not exist: {args.data_dir}\n"
            "       Download GRABMyo first (see instructions above) then pass\n"
            "       the download root directory to --data-dir.",
            file=sys.stderr,
        )
        return 1

    if not args.data_dir.is_dir():
        print(
            f"ERROR: --data-dir is not a directory: {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    if args.max_subjects < 1:
        print(
            f"ERROR: --max-subjects must be >= 1 (got {args.max_subjects}).",
            file=sys.stderr,
        )
        return 1

    # Run the conversion.
    try:
        convert_grabmyo_dataset(
            grabmyo_dir=args.data_dir,
            output_dir=args.output_dir,
            max_subjects=args.max_subjects,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"ERROR: Missing dependency — {exc}\n"
            "       Install required packages with:  pip install wfdb scipy pandas numpy",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: Unexpected failure during conversion: {exc}", file=sys.stderr)
        raise  # re-raise so the full traceback is visible for debugging

    print(
        f"\nConversion complete.  Session CSVs written to: {args.output_dir.resolve()}\n"
        f"You can now load them with:\n"
        f"    from src.data.loader import load_dataset\n"
        f"    df = load_dataset('{args.output_dir}')\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

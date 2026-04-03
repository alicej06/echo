"""
Download Meta Research EMG datasets.

Datasets:
  emg2pose      — 193 participants, 370 hours, hand joint angles
                  https://github.com/facebookresearch/emg2pose
  gni           — generic-neuromotor-interface, 100 participants, 200+ hours
                  https://github.com/facebookresearch/generic-neuromotor-interface
  emg2qwerty    — 108 users, 346 hours, bilateral wrist EMG typing
                  https://github.com/facebookresearch/emg2qwerty

All datasets are CC-BY-NC-4.0. You must request dataset access before downloading.
Follow the instructions at each GitHub repo to obtain credentials/download URLs.

Usage:
    python scripts/download_meta_emg_data.py --dataset emg2pose --output data/meta/
    python scripts/download_meta_emg_data.py --dataset gni --output data/meta/
    python scripts/download_meta_emg_data.py --dataset all --output data/meta/
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path


DATASET_REPOS = {
    "emg2pose": {
        "github": "https://github.com/facebookresearch/emg2pose",
        "description": "EMG → 63-DoF hand joint angles, 193 participants, 370 hours",
        "download_note": (
            "Request dataset access at https://github.com/facebookresearch/emg2pose\n"
            "Then set env var EMG2POSE_DOWNLOAD_URL and re-run this script."
        ),
        "env_var": "EMG2POSE_DOWNLOAD_URL",
    },
    "gni": {
        "github": "https://github.com/facebookresearch/generic-neuromotor-interface",
        "description": "Discrete gestures + handwriting + wrist, 100 participants, 200+ hours (Nature 2025)",
        "download_note": (
            "Request dataset access at https://github.com/facebookresearch/generic-neuromotor-interface\n"
            "Then set env var GNI_DOWNLOAD_URL and re-run this script."
        ),
        "env_var": "GNI_DOWNLOAD_URL",
    },
    "emg2qwerty": {
        "github": "https://github.com/facebookresearch/emg2qwerty",
        "description": "Bilateral wrist EMG typing, 108 users, 346 hours",
        "download_note": (
            "Request dataset access at https://github.com/facebookresearch/emg2qwerty\n"
            "Then set env var EMG2QWERTY_DOWNLOAD_URL and re-run this script."
        ),
        "env_var": "EMG2QWERTY_DOWNLOAD_URL",
    },
}


def download_dataset(name: str, output_dir: Path) -> None:
    info = DATASET_REPOS[name]
    dest = output_dir / name
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Description: {info['description']}")
    print(f"GitHub: {info['github']}")
    print(f"{'='*60}")

    url = os.environ.get(info["env_var"])
    if not url:
        print(f"\n[BLOCKED] Download URL not set.")
        print(f"\nTo get access:\n  {info['download_note']}")
        print(f"\nOnce you have the URL, run:")
        print(f"  {info['env_var']}=<url> python scripts/download_meta_emg_data.py --dataset {name}")
        return

    print(f"Downloading from: {url}")
    print(f"Output: {dest}/")

    # Use wget with progress and resume support
    cmd = ["wget", "-c", "--progress=dot:giga", "-P", str(dest), url]
    try:
        subprocess.run(cmd, check=True)
        print(f"\nDownload complete: {dest}/")
        _print_next_steps(name, dest)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        # Fall back to curl
        print("wget not found, trying curl...")
        filename = url.split("/")[-1]
        cmd = ["curl", "-L", "-C", "-", "-o", str(dest / filename), url]
        subprocess.run(cmd, check=True)
        print(f"\nDownload complete: {dest / filename}")
        _print_next_steps(name, dest)


def _print_next_steps(name: str, dest: Path) -> None:
    print(f"\nNext steps for {name}:")
    if name == "emg2pose":
        print(f"  python scripts/pretrain_on_meta_data.py --task pose --data {dest}")
    elif name == "gni":
        print(f"  python scripts/pretrain_on_meta_data.py --task gestures --data {dest}")
    elif name == "emg2qwerty":
        print(f"  python scripts/pretrain_on_meta_data.py --task qwerty --data {dest}")
    print(f"\nOr load directly with:")
    print(f"  from src.data.meta_loader import load_meta_windows")
    print(f"  X, y = load_meta_windows('{dest}', dataset='{name}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Meta Research EMG datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REPOS.keys()) + ["all"],
        default="emg2pose",
        help="Which dataset to download (default: emg2pose)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta"),
        help="Output directory (default: data/meta/)",
    )
    args = parser.parse_args()

    targets = list(DATASET_REPOS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"MAIA EMG-ASL — Meta Dataset Downloader")
    print(f"Output root: {args.output.resolve()}")
    print(f"Datasets: {', '.join(targets)}")
    print()
    print("NOTE: All Meta datasets are CC-BY-NC-4.0.")
    print("You must request access before downloading. See each dataset's GitHub page.")

    for name in targets:
        download_dataset(name, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()

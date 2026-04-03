#!/usr/bin/env python3
"""
Download trained model artifacts from Cloudflare R2.

Used by:
  - Railway server startup (via R2_MODEL_URL env var)
  - Developers syncing latest model to local machine
  - SLURM jobs that need the latest model for evaluation

Usage:
  # Download the latest LSTM model:
  python scripts/download_artifact.py --model asl_emg_classifier --tag latest

  # Download a specific version:
  python scripts/download_artifact.py --model conformer_classifier --tag v1.2.0

  # Download all latest models:
  python scripts/download_artifact.py --all-latest

  # Just print the public URL (for setting R2_MODEL_URL in Railway):
  python scripts/download_artifact.py --model asl_emg_classifier --url-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the shared helpers from upload_artifact so credential checking,
# client construction, and SHA256 verification stay in one place.
from upload_artifact import (
    _check_credentials,
    _make_client,
    download_artifact,
    generate_public_url,
    list_artifacts,
)

# ---------------------------------------------------------------------------
# Known model names (used for --all-latest)
# ---------------------------------------------------------------------------

KNOWN_MODELS: list[str] = [
    "asl_emg_classifier",
    "conformer_classifier",
    "cross_modal_asl",
]


# ---------------------------------------------------------------------------
# URL-only helper
# ---------------------------------------------------------------------------

def _get_public_url_for_model(
    model_name: str,
    tag: str,
    cfg: dict[str, str],
) -> str:
    """
    Resolve the public R2 URL for a model without downloading it.
    Lists the bucket prefix to discover the filename, then calls
    generate_public_url().
    """
    client = _make_client(cfg)
    bucket = cfg["R2_BUCKET_NAME"]

    prefix = f"models/{model_name}/{tag}/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])

    model_key = next(
        (obj["Key"] for obj in contents if not obj["Key"].endswith(".sha256")),
        None,
    )

    if model_key is None:
        print(f"ERROR: No artifact found at s3://{bucket}/{prefix}")
        print("  Check available artifacts with:")
        print("    python scripts/upload_artifact.py --list")
        sys.exit(1)

    url = generate_public_url(model_key, cfg)
    if not url:
        print(
            "ERROR: R2_PUBLIC_DOMAIN is not set.\n"
            "  Enable the public development URL on your R2 bucket, then:\n"
            "    export R2_PUBLIC_DOMAIN=pub-<hash>.r2.dev\n"
            "  Or configure a custom domain and set R2_PUBLIC_DOMAIN to it."
        )
        sys.exit(1)
    return url


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EMG-ASL model artifacts from Cloudflare R2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--model",
        metavar="NAME",
        help=(
            "Model name to download, e.g. asl_emg_classifier, "
            "conformer_classifier, cross_modal_asl."
        ),
    )
    action.add_argument(
        "--all-latest",
        action="store_true",
        help=(
            f"Download the latest version of all known models: "
            f"{', '.join(KNOWN_MODELS)}."
        ),
    )
    action.add_argument(
        "--list",
        action="store_true",
        help="List all artifacts stored in R2 without downloading.",
    )

    parser.add_argument(
        "--tag",
        metavar="TAG",
        default="latest",
        help="Version tag to download. Defaults to 'latest'.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="models/",
        help="Local directory to write downloaded files. Defaults to models/.",
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help=(
            "Print the public R2 URL for the artifact instead of downloading it. "
            "Useful for setting R2_MODEL_URL in Railway."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _check_credentials()

    # --list mode
    if args.list:
        list_artifacts(cfg)
        return

    # --all-latest mode
    if args.all_latest:
        for model_name in KNOWN_MODELS:
            print(f"[download] Model: {model_name} (tag=latest)")
            try:
                download_artifact(
                    model_name=model_name,
                    tag="latest",
                    output_dir=args.output_dir,
                    cfg=cfg,
                )
            except SystemExit:
                print(
                    f"[download] WARNING: Could not download {model_name}. "
                    "It may not have been uploaded yet. Continuing...\n"
                )
        return

    # Single-model mode
    model_name = args.model

    if args.url_only:
        url = _get_public_url_for_model(model_name, args.tag, cfg)
        print(url)
        return

    out_path = download_artifact(
        model_name=model_name,
        tag=args.tag,
        output_dir=args.output_dir,
        cfg=cfg,
    )
    print(f"[download] Done: {out_path.resolve()}")


if __name__ == "__main__":
    main()

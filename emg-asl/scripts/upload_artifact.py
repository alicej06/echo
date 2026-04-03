#!/usr/bin/env python3
"""
Upload trained EMG-ASL model artifacts to Cloudflare R2.

R2 is S3-compatible with no egress fees. Free tier: 10 GB storage.

Setup (one-time):
  1. Create a Cloudflare account at cloudflare.com
  2. Go to R2 -> Create bucket -> name it 'maia-emg-asl'
  3. Create an API token with R2 read+write permissions
  4. Set environment variables:
       export R2_ACCOUNT_ID=your-account-id
       export R2_ACCESS_KEY_ID=your-access-key-id
       export R2_SECRET_ACCESS_KEY=your-secret-key
       export R2_BUCKET_NAME=maia-emg-asl

Usage:
  # Upload a specific model:
  python scripts/upload_artifact.py --file models/asl_emg_classifier.onnx

  # Upload all models in models/ directory:
  python scripts/upload_artifact.py --all

  # Upload with a version tag:
  python scripts/upload_artifact.py --file models/conformer_classifier.onnx --tag v1.2.0

  # List all artifacts in R2:
  python scripts/upload_artifact.py --list

  # Set as 'latest' (updates the latest pointer):
  python scripts/upload_artifact.py --file models/conformer_classifier.onnx --set-latest
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

REQUIRED_ENV_VARS = [
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
]

OPTIONAL_ENV_VARS = {
    "R2_BUCKET_NAME": "maia-emg-asl",
    "R2_PUBLIC_DOMAIN": "",  # e.g. pub-<hash>.r2.dev or custom domain
}


def _check_credentials() -> dict[str, str]:
    """
    Validate that all required R2 environment variables are present.
    Prints setup instructions and exits with code 1 if any are missing.
    Returns a dict of all resolved config values.
    """
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        print("ERROR: Missing Cloudflare R2 credentials.\n")
        print("Missing environment variables:")
        for var in missing:
            print(f"  {var}")
        print(
            "\nSetup instructions:\n"
            "  1. Log in to https://dash.cloudflare.com\n"
            "  2. Navigate to R2 -> Overview -> Create bucket\n"
            "     Name your bucket 'maia-emg-asl' (or set R2_BUCKET_NAME)\n"
            "  3. Go to R2 -> Manage R2 API Tokens -> Create API Token\n"
            "     Select 'Object Read & Write' permissions\n"
            "  4. Copy Account ID from the R2 Overview page\n"
            "  5. Export the variables before running this script:\n"
            "\n"
            "     export R2_ACCOUNT_ID=your-account-id\n"
            "     export R2_ACCESS_KEY_ID=your-access-key-id\n"
            "     export R2_SECRET_ACCESS_KEY=your-secret-key\n"
            "     export R2_BUCKET_NAME=maia-emg-asl   # optional, default shown\n"
        )
        sys.exit(1)

    cfg: dict[str, str] = {v: os.environ[v] for v in REQUIRED_ENV_VARS}
    for var, default in OPTIONAL_ENV_VARS.items():
        cfg[var] = os.environ.get(var, default)
    return cfg


def _make_client(cfg: dict[str, str]):
    """Return a boto3 S3 client pointed at the Cloudflare R2 endpoint."""
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 is not installed. Run: pip install boto3")
        sys.exit(1)

    endpoint = f"https://{cfg['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=cfg["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=cfg["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    return client


# ---------------------------------------------------------------------------
# SHA256 helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(file_path: Path) -> str:
    """Compute the hex SHA256 digest of a local file."""
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Progress callback factory
# ---------------------------------------------------------------------------

def _make_progress_callback(file_size: int, label: str):
    """
    Return a boto3 upload/download progress callback that prints a
    tqdm-style progress bar to stdout.
    """
    transferred = [0]

    def _callback(n_bytes: int) -> None:
        transferred[0] += n_bytes
        pct = transferred[0] / file_size * 100 if file_size > 0 else 100
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        mb_done = transferred[0] / (1 << 20)
        mb_total = file_size / (1 << 20)
        print(
            f"\r  {label}: [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
            end="",
            flush=True,
        )
        if transferred[0] >= file_size:
            print()  # newline after completion

    return _callback


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def upload_artifact(
    file_path: str | Path,
    tag: str | None = None,
    set_latest: bool = False,
    cfg: dict[str, str] | None = None,
) -> str:
    """
    Upload a model artifact to Cloudflare R2.

    The file is stored at:
        models/{model_name}/{tag}/{filename}

    A SHA256 sidecar file is stored alongside it at:
        models/{model_name}/{tag}/{filename}.sha256

    If set_latest=True, the file is also copied (overwritten) at:
        models/{model_name}/latest/{filename}
        models/{model_name}/latest/{filename}.sha256

    Parameters
    ----------
    file_path:
        Local path to the model file to upload.
    tag:
        Version tag string. Defaults to a UTC timestamp (YYYYMMDD_HHMMSS).
    set_latest:
        When True, also write to the 'latest' prefix so the Railway server
        can pull the newest model by pointing at a stable URL.
    cfg:
        Pre-loaded config dict (from _check_credentials). If None, credentials
        are loaded from the environment.

    Returns
    -------
    The R2 object key for the versioned upload (not the 'latest' alias).
    """
    if cfg is None:
        cfg = _check_credentials()

    client = _make_client(cfg)
    bucket = cfg["R2_BUCKET_NAME"]

    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    model_name = path.stem          # e.g. 'asl_emg_classifier'
    filename = path.name            # e.g. 'asl_emg_classifier.onnx'
    file_size = path.stat().st_size

    if tag is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    versioned_prefix = f"models/{model_name}/{tag}"
    versioned_key = f"{versioned_prefix}/{filename}"
    sha_key = f"{versioned_key}.sha256"

    print(f"[upload] Computing SHA256 for {filename} ({file_size / (1<<20):.1f} MB)...")
    digest = _sha256_of_file(path)
    print(f"[upload] SHA256: {digest}")

    print(f"[upload] Uploading to s3://{bucket}/{versioned_key}")
    callback = _make_progress_callback(file_size, "upload")
    with path.open("rb") as fh:
        client.upload_fileobj(fh, bucket, versioned_key, Callback=callback)

    digest_bytes = digest.encode()
    client.put_object(Bucket=bucket, Key=sha_key, Body=digest_bytes)
    print(f"[upload] SHA256 sidecar written to s3://{bucket}/{sha_key}")

    if set_latest:
        latest_prefix = f"models/{model_name}/latest"
        latest_key = f"{latest_prefix}/{filename}"
        latest_sha_key = f"{latest_key}.sha256"

        print(f"[upload] Updating latest pointer -> s3://{bucket}/{latest_key}")
        callback2 = _make_progress_callback(file_size, "latest")
        with path.open("rb") as fh:
            client.upload_fileobj(fh, bucket, latest_key, Callback=callback2)
        client.put_object(Bucket=bucket, Key=latest_sha_key, Body=digest_bytes)
        print(f"[upload] Latest SHA256 sidecar written.")

    public_url = generate_public_url(versioned_key, cfg)
    if public_url:
        print(f"[upload] Public URL: {public_url}")
    else:
        print(f"[upload] R2 key: {versioned_key}")
        print("[upload] Note: set R2_PUBLIC_DOMAIN to get a public URL.")

    return versioned_key


def download_artifact(
    model_name: str,
    tag: str = "latest",
    output_dir: str = "models/",
    cfg: dict[str, str] | None = None,
) -> Path:
    """
    Download a model artifact from Cloudflare R2.

    Downloads from:
        models/{model_name}/{tag}/{filename}
    and verifies the SHA256 sidecar after download.

    Parameters
    ----------
    model_name:
        The model stem name, e.g. 'asl_emg_classifier'.
    tag:
        Version tag or 'latest'.
    output_dir:
        Local directory to write the downloaded file.
    cfg:
        Pre-loaded credential config. Loaded from environment if None.

    Returns
    -------
    The local Path where the file was written.
    """
    if cfg is None:
        cfg = _check_credentials()

    client = _make_client(cfg)
    bucket = cfg["R2_BUCKET_NAME"]

    # List objects under this model/tag prefix to discover the filename.
    prefix = f"models/{model_name}/{tag}/"
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get("Contents", [])

    if not contents:
        print(f"ERROR: No artifacts found at s3://{bucket}/{prefix}")
        print(f"  Run: python scripts/upload_artifact.py --list")
        sys.exit(1)

    # Find the main model file (not the .sha256 sidecar).
    model_key = next(
        (obj["Key"] for obj in contents if not obj["Key"].endswith(".sha256")),
        None,
    )
    sha_key = next(
        (obj["Key"] for obj in contents if obj["Key"].endswith(".sha256")),
        None,
    )

    if model_key is None:
        print(f"ERROR: No model file found under s3://{bucket}/{prefix}")
        sys.exit(1)

    filename = Path(model_key).name
    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get file size for progress bar.
    head = client.head_object(Bucket=bucket, Key=model_key)
    file_size = head["ContentLength"]

    print(f"[download] Fetching s3://{bucket}/{model_key} ({file_size / (1<<20):.1f} MB)")
    callback = _make_progress_callback(file_size, "download")
    with out_path.open("wb") as fh:
        client.download_fileobj(bucket, model_key, fh, Callback=callback)

    print(f"[download] Saved to {out_path}")

    if sha_key:
        expected_obj = client.get_object(Bucket=bucket, Key=sha_key)
        expected_digest = expected_obj["Body"].read().decode().strip()
        print(f"[download] Verifying SHA256 (expected: {expected_digest[:16]}...)...")
        actual_digest = _sha256_of_file(out_path)
        if actual_digest != expected_digest:
            print(
                f"ERROR: SHA256 mismatch!\n"
                f"  expected : {expected_digest}\n"
                f"  actual   : {actual_digest}\n"
                f"The file may be corrupt. Delete {out_path} and retry."
            )
            sys.exit(1)
        print(f"[download] SHA256 OK: {actual_digest[:16]}...")
    else:
        print("[download] No SHA256 sidecar found; skipping integrity check.")

    return out_path


def list_artifacts(cfg: dict[str, str] | None = None) -> None:
    """
    Print all objects in the R2 bucket with size and last-modified date.
    """
    if cfg is None:
        cfg = _check_credentials()

    client = _make_client(cfg)
    bucket = cfg["R2_BUCKET_NAME"]

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket)

    total_bytes = 0
    count = 0

    print(f"\nArtifacts in s3://{bucket}/\n")
    print(f"  {'Key':<70}  {'Size':>10}  {'Last Modified'}")
    print(f"  {'-'*70}  {'-'*10}  {'-'*20}")

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size_mb = obj["Size"] / (1 << 20)
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M UTC")
            print(f"  {key:<70}  {size_mb:>9.2f}M  {modified}")
            total_bytes += obj["Size"]
            count += 1

    if count == 0:
        print("  (no objects found)")
    else:
        print(f"\n  {count} objects, {total_bytes / (1<<20):.1f} MB total")


def generate_public_url(key: str, cfg: dict[str, str] | None = None) -> str:
    """
    Generate the public HTTPS URL for an R2 object key.

    Requires either:
    - R2_PUBLIC_DOMAIN set to the bucket's public dev URL
      (e.g. 'pub-<hash>.r2.dev' or a custom domain), OR
    - The bucket to have the public development URL feature enabled.

    Returns an empty string if R2_PUBLIC_DOMAIN is not configured.
    """
    if cfg is None:
        cfg = _check_credentials()

    domain = cfg.get("R2_PUBLIC_DOMAIN", "").strip()
    if not domain:
        return ""
    return f"https://{domain}/{key}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload EMG-ASL model artifacts to Cloudflare R2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--file",
        metavar="PATH",
        help="Upload a single model file.",
    )
    action.add_argument(
        "--all",
        action="store_true",
        help="Upload all .onnx and .pt files found under models/.",
    )
    action.add_argument(
        "--list",
        action="store_true",
        help="List all artifacts stored in R2.",
    )

    parser.add_argument(
        "--tag",
        metavar="TAG",
        default=None,
        help=(
            "Version tag to attach to this upload (e.g. v1.2.0 or slurm_12345). "
            "Defaults to a UTC timestamp."
        ),
    )
    parser.add_argument(
        "--set-latest",
        action="store_true",
        help="Also overwrite the 'latest' prefix in R2 with this file.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list:
        cfg = _check_credentials()
        list_artifacts(cfg)
        return

    cfg = _check_credentials()

    if args.file:
        upload_artifact(args.file, tag=args.tag, set_latest=args.set_latest, cfg=cfg)
        return

    if args.all:
        model_dir = Path("models")
        files = sorted(
            list(model_dir.rglob("*.onnx")) + list(model_dir.rglob("*.pt"))
        )
        if not files:
            print(f"No .onnx or .pt files found under {model_dir.resolve()}")
            sys.exit(1)
        print(f"Found {len(files)} model file(s) to upload:")
        for f in files:
            print(f"  {f}")
        print()
        for f in files:
            upload_artifact(f, tag=args.tag, set_latest=args.set_latest, cfg=cfg)
            print()


if __name__ == "__main__":
    main()

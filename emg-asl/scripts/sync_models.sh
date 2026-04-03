#!/usr/bin/env bash
# sync_models.sh -- Sync trained model artifacts with Cloudflare R2.
#
# Usage:
#   ./scripts/sync_models.sh           # download latest from R2
#   ./scripts/sync_models.sh --upload  # upload local models to R2
#   ./scripts/sync_models.sh --status  # show sync status without downloading
#
# Color legend:
#   GREEN  -- file is in sync (same SHA256 locally and in R2)
#   YELLOW -- R2 is newer than local (should download)
#   RED    -- local is newer than R2 (should upload)
#   CYAN   -- file exists only in R2 (not downloaded yet)
#   GREY   -- file exists only locally (not uploaded yet)
#
# Requirements:
#   pip install boto3
#   export R2_ACCOUNT_ID=...
#   export R2_ACCESS_KEY_ID=...
#   export R2_SECRET_ACCESS_KEY=...
#   export R2_BUCKET_NAME=maia-emg-asl   # default if not set
#
# The script uses Python for S3 operations (via upload_artifact.py helpers)
# and calls the AWS CLI only if boto3 is unavailable.

set -euo pipefail

# ---------------------------------------------------------------------------
# Terminal color codes
# ---------------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
GREY='\033[0;90m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_MODEL_DIR="${PROJECT_ROOT}/models"
PYTHON="${PYTHON:-python}"

# Known model names mirroring KNOWN_MODELS in download_artifact.py.
KNOWN_MODELS=(
    "asl_emg_classifier"
    "conformer_classifier"
    "cross_modal_asl"
)

# File extensions to consider.
MODEL_EXTENSIONS=("onnx" "pt")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

MODE="download"  # default

for arg in "$@"; do
    case "$arg" in
        --upload)  MODE="upload"  ;;
        --status)  MODE="status"  ;;
        --help|-h)
            echo "Usage: $0 [--upload | --status]"
            echo ""
            echo "  (no flag)  Download all latest models from R2 to models/"
            echo "  --upload   Upload all local .onnx/.pt files to R2 as 'latest'"
            echo "  --status   Compare local vs R2 without transferring any files"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

_check_env() {
    local missing=0
    for var in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
        if [ -z "${!var:-}" ]; then
            echo -e "${RED}ERROR: ${var} is not set.${RESET}"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        echo ""
        echo "Set your Cloudflare R2 credentials before running this script:"
        echo "  export R2_ACCOUNT_ID=your-account-id"
        echo "  export R2_ACCESS_KEY_ID=your-access-key-id"
        echo "  export R2_SECRET_ACCESS_KEY=your-secret-key"
        echo "  export R2_BUCKET_NAME=maia-emg-asl"
        exit 1
    fi
}

_check_env

# Resolve bucket name (use default if not set).
R2_BUCKET_NAME="${R2_BUCKET_NAME:-maia-emg-asl}"
export R2_BUCKET_NAME

# ---------------------------------------------------------------------------
# Python helper: compute SHA256 of a local file
# ---------------------------------------------------------------------------

_local_sha256() {
    local file="$1"
    "$PYTHON" - "$file" <<'PYEOF'
import hashlib, sys
path = sys.argv[1]
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
PYEOF
}

# ---------------------------------------------------------------------------
# Python helper: fetch SHA256 sidecar from R2 for a given model/tag
# Returns empty string if not found.
# ---------------------------------------------------------------------------

_r2_sha256() {
    local model_name="$1"
    local tag="${2:-latest}"
    "$PYTHON" - "$model_name" "$tag" <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
# Running from project root; add scripts/ to path so import works.
import importlib.util, pathlib

scripts_dir = pathlib.Path(__file__).parent / "scripts" if False else pathlib.Path("scripts")

# Inline the credential + client logic to avoid path issues when called
# from a subprocess.
import boto3

model_name = sys.argv[1]
tag = sys.argv[2]
account_id = os.environ["R2_ACCOUNT_ID"]
access_key = os.environ["R2_ACCESS_KEY_ID"]
secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
bucket = os.environ.get("R2_BUCKET_NAME", "maia-emg-asl")

client = boto3.client(
    "s3",
    endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto",
)

prefix = f"models/{model_name}/{tag}/"
try:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])
    sha_key = next((o["Key"] for o in contents if o["Key"].endswith(".sha256")), None)
    if sha_key:
        obj = client.get_object(Bucket=bucket, Key=sha_key)
        print(obj["Body"].read().decode().strip())
    else:
        print("")
except Exception:
    print("")
PYEOF
}

# ---------------------------------------------------------------------------
# Python helper: get last-modified timestamp (epoch) of R2 object
# Returns 0 if not found.
# ---------------------------------------------------------------------------

_r2_mtime() {
    local model_name="$1"
    local tag="${2:-latest}"
    "$PYTHON" - "$model_name" "$tag" <<'PYEOF'
import sys, os, boto3
from datetime import timezone

model_name = sys.argv[1]
tag = sys.argv[2]
account_id = os.environ["R2_ACCOUNT_ID"]
access_key = os.environ["R2_ACCESS_KEY_ID"]
secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
bucket = os.environ.get("R2_BUCKET_NAME", "maia-emg-asl")

client = boto3.client(
    "s3",
    endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="auto",
)

prefix = f"models/{model_name}/{tag}/"
try:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = [o for o in resp.get("Contents", []) if not o["Key"].endswith(".sha256")]
    if contents:
        ts = contents[0]["LastModified"].astimezone(timezone.utc).timestamp()
        print(int(ts))
    else:
        print(0)
except Exception:
    print(0)
PYEOF
}

# ---------------------------------------------------------------------------
# Status computation
# ---------------------------------------------------------------------------

declare -A STATUS       # model -> status string
declare -A LOCAL_SHA    # model -> local sha256
declare -A R2_SHA       # model -> r2 sha256
declare -A R2_MTIME     # model -> r2 mtime (epoch)
declare -A LOCAL_PATH   # model -> local file path

echo ""
echo -e "${BOLD}Scanning local models directory: ${LOCAL_MODEL_DIR}${RESET}"
echo -e "${BOLD}R2 bucket: ${R2_BUCKET_NAME}${RESET}"
echo ""

for model_name in "${KNOWN_MODELS[@]}"; do
    # Find the local file for this model (prefer .onnx over .pt).
    local_file=""
    for ext in "${MODEL_EXTENSIONS[@]}"; do
        candidate="${LOCAL_MODEL_DIR}/${model_name}.${ext}"
        if [ -f "$candidate" ]; then
            local_file="$candidate"
            break
        fi
    done
    LOCAL_PATH["$model_name"]="$local_file"

    echo -n "  Checking ${model_name}..."

    r2_digest=$(_r2_sha256 "$model_name" "latest")
    r2_ts=$(_r2_mtime "$model_name" "latest")
    R2_SHA["$model_name"]="$r2_digest"
    R2_MTIME["$model_name"]="$r2_ts"

    if [ -z "$local_file" ] && [ -z "$r2_digest" ]; then
        STATUS["$model_name"]="MISSING_BOTH"
    elif [ -z "$local_file" ] && [ -n "$r2_digest" ]; then
        STATUS["$model_name"]="R2_ONLY"
    elif [ -n "$local_file" ] && [ -z "$r2_digest" ]; then
        local_digest=$(_local_sha256 "$local_file")
        LOCAL_SHA["$model_name"]="$local_digest"
        STATUS["$model_name"]="LOCAL_ONLY"
    else
        local_digest=$(_local_sha256 "$local_file")
        LOCAL_SHA["$model_name"]="$local_digest"
        if [ "$local_digest" = "$r2_digest" ]; then
            STATUS["$model_name"]="IN_SYNC"
        else
            local_mtime=$(date -r "$local_file" +%s 2>/dev/null || stat -f %m "$local_file" 2>/dev/null || echo 0)
            if [ "$r2_ts" -gt "$local_mtime" ]; then
                STATUS["$model_name"]="R2_NEWER"
            else
                STATUS["$model_name"]="LOCAL_NEWER"
            fi
        fi
    fi

    echo " done"
done

# ---------------------------------------------------------------------------
# Print status table
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}  Model                        Status          Local SHA256        R2 SHA256${RESET}"
echo "  $(printf '%-28s' '---') $(printf '%-15s' '------') $(printf '%-18s' '----------') ----------"

for model_name in "${KNOWN_MODELS[@]}"; do
    st="${STATUS[$model_name]}"
    local_sha="${LOCAL_SHA[$model_name]:-}"
    r2_sha="${R2_SHA[$model_name]:-}"
    local_sha_short="${local_sha:0:12}..."
    r2_sha_short="${r2_sha:0:12}..."
    [ -z "$local_sha" ] && local_sha_short="(none)"
    [ -z "$r2_sha" ]    && r2_sha_short="(none)"

    case "$st" in
        IN_SYNC)
            color="${GREEN}"
            label="IN SYNC"
            ;;
        R2_NEWER)
            color="${YELLOW}"
            label="R2 NEWER"
            ;;
        LOCAL_NEWER)
            color="${RED}"
            label="LOCAL NEWER"
            ;;
        R2_ONLY)
            color="${CYAN}"
            label="R2 ONLY"
            ;;
        LOCAL_ONLY)
            color="${GREY}"
            label="LOCAL ONLY"
            ;;
        MISSING_BOTH)
            color="${GREY}"
            label="NOT FOUND"
            ;;
        *)
            color="${RESET}"
            label="UNKNOWN"
            ;;
    esac

    printf "  ${color}%-28s %-15s %-18s %s${RESET}\n" \
        "$model_name" "$label" "$local_sha_short" "$r2_sha_short"
done

echo ""

# ---------------------------------------------------------------------------
# Early exit for --status mode
# ---------------------------------------------------------------------------

if [ "$MODE" = "status" ]; then
    echo "Status check complete. No files were transferred."
    exit 0
fi

# ---------------------------------------------------------------------------
# Download mode: pull R2-newer and R2-only models
# ---------------------------------------------------------------------------

if [ "$MODE" = "download" ]; then
    downloaded=0
    skipped=0
    for model_name in "${KNOWN_MODELS[@]}"; do
        st="${STATUS[$model_name]}"
        if [ "$st" = "R2_NEWER" ] || [ "$st" = "R2_ONLY" ]; then
            echo -e "${YELLOW}[sync] Downloading ${model_name} (${st})...${RESET}"
            cd "${PROJECT_ROOT}"
            "$PYTHON" scripts/download_artifact.py \
                --model "$model_name" \
                --tag latest \
                --output-dir models/
            downloaded=$((downloaded + 1))
        elif [ "$st" = "IN_SYNC" ]; then
            echo -e "${GREEN}[sync] ${model_name} is already up to date.${RESET}"
            skipped=$((skipped + 1))
        elif [ "$st" = "LOCAL_NEWER" ]; then
            echo -e "${RED}[sync] ${model_name}: local is newer. Skipping (use --upload to push to R2).${RESET}"
            skipped=$((skipped + 1))
        elif [ "$st" = "LOCAL_ONLY" ]; then
            echo -e "${GREY}[sync] ${model_name}: exists only locally. Skipping (use --upload to push to R2).${RESET}"
            skipped=$((skipped + 1))
        else
            echo -e "${GREY}[sync] ${model_name}: not found locally or in R2. Skipping.${RESET}"
            skipped=$((skipped + 1))
        fi
    done
    echo ""
    echo "[sync] Download complete. ${downloaded} downloaded, ${skipped} skipped."
fi

# ---------------------------------------------------------------------------
# Upload mode: push local-newer and local-only models
# ---------------------------------------------------------------------------

if [ "$MODE" = "upload" ]; then
    uploaded=0
    skipped=0
    for model_name in "${KNOWN_MODELS[@]}"; do
        st="${STATUS[$model_name]}"
        local_file="${LOCAL_PATH[$model_name]:-}"
        if [ "$st" = "LOCAL_NEWER" ] || [ "$st" = "LOCAL_ONLY" ]; then
            echo -e "${RED}[sync] Uploading ${model_name} (${st})...${RESET}"
            cd "${PROJECT_ROOT}"
            "$PYTHON" scripts/upload_artifact.py \
                --file "$local_file" \
                --set-latest
            uploaded=$((uploaded + 1))
        elif [ "$st" = "IN_SYNC" ]; then
            echo -e "${GREEN}[sync] ${model_name} is already up to date. Skipping.${RESET}"
            skipped=$((skipped + 1))
        elif [ "$st" = "R2_NEWER" ]; then
            echo -e "${YELLOW}[sync] ${model_name}: R2 is newer. Skipping (use default mode to download first).${RESET}"
            skipped=$((skipped + 1))
        elif [ "$st" = "R2_ONLY" ]; then
            echo -e "${CYAN}[sync] ${model_name}: exists only in R2, not locally. Skipping.${RESET}"
            skipped=$((skipped + 1))
        else
            echo -e "${GREY}[sync] ${model_name}: not found locally or in R2. Skipping.${RESET}"
            skipped=$((skipped + 1))
        fi
    done
    echo ""
    echo "[sync] Upload complete. ${uploaded} uploaded, ${skipped} skipped."
fi

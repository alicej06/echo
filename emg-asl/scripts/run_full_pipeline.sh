#!/usr/bin/env bash
# run_full_pipeline.sh
#
# Runs the complete EMG-ASL pipeline end-to-end:
#   1. Install dependencies
#   2. Download Italian SL dataset (~50 MB, no account needed)
#   3. Train LSTM on Italian SL data (CPU, ~5-10 min)
#   4. Run smoke tests (8/8 must pass)
#   5. Start inference server
#
# Usage:
#   ./scripts/run_full_pipeline.sh              # full pipeline
#   ./scripts/run_full_pipeline.sh --skip-train # use existing model
#   ./scripts/run_full_pipeline.sh --gpu        # train on GPU if available
#
# Expected output:
#   [1/5] Dependencies: OK
#   [2/5] Italian SL data: downloaded (47 MB)
#   [3/5] Training: 85.3% accuracy (50 epochs, 4m 23s)
#   [4/5] Smoke tests: 8/8 PASS
#   [5/5] Server: http://localhost:8000 (ws://localhost:8000/stream)

set -euo pipefail

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'   # reset

ok()      { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}$*${NC}"; }
step()    { echo -e "${BOLD}$*${NC}"; }

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

SKIP_TRAIN=false
USE_GPU=false

for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=true ;;
        --gpu)        USE_GPU=true ;;
        --help|-h)
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            err "Unknown argument: $arg  (use --skip-train, --gpu, or --help)"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve project root (script may be called from any CWD)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# STEP 1 -- Dependencies
# ---------------------------------------------------------------------------

header "[1/5] Checking dependencies..."

# Python version check (>= 3.11)
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        _ver="$("$candidate" -c 'import sys; print(sys.version_info[:2])'  2>/dev/null || true)"
        _major="$("$candidate" -c 'import sys; print(sys.version_info[0])' 2>/dev/null || true)"
        _minor="$("$candidate" -c 'import sys; print(sys.version_info[1])' 2>/dev/null || true)"
        if [[ "$_major" -ge 3 && "$_minor" -ge 11 ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    err "Python >= 3.11 not found. Install it from https://python.org and retry."
fi
ok "Python: $($PYTHON_BIN --version)"

# Virtual environment
VENV_DIR="${PROJECT_ROOT}/venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    step "Creating virtual environment at ${VENV_DIR}..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# Activate venv for the rest of this script
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
ok "Virtual environment active: $(which python)"

# Install requirements only if core packages are missing
if ! python -c "import torch, fastapi, numpy" &>/dev/null 2>&1; then
    step "Installing requirements (first run -- this may take a few minutes)..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "${PROJECT_ROOT}/requirements.txt"
    ok "Requirements installed."
else
    ok "Requirements already satisfied (skipping pip install)."
fi

echo -e "${GREEN}[1/5] Dependencies: OK${NC}"

# ---------------------------------------------------------------------------
# STEP 2 -- Italian SL dataset
# ---------------------------------------------------------------------------

header "[2/5] Checking Italian SL dataset..."

ITALIAN_SL_RAW="${PROJECT_ROOT}/data/raw/italian_sl"
ITALIAN_SL_REPO="${PROJECT_ROOT}/data/italian_sl_repo"

# Count existing converted CSVs
existing_csvs=0
if [[ -d "$ITALIAN_SL_RAW" ]]; then
    existing_csvs="$(find "$ITALIAN_SL_RAW" -name "ISL*.csv" 2>/dev/null | wc -l | tr -d ' ')"
fi

if [[ "$existing_csvs" -gt 0 ]]; then
    ok "Italian SL data already present (${existing_csvs} session CSVs in ${ITALIAN_SL_RAW})."
    echo -e "${GREEN}[2/5] Italian SL data: already present (${existing_csvs} CSVs)${NC}"
else
    # Check git is available
    if ! command -v git &>/dev/null; then
        err "git is required to download the Italian SL dataset. Install git and retry."
    fi

    step "Cloning Italian SL repository (~50 MB)..."
    if [[ -d "$ITALIAN_SL_REPO" ]]; then
        warn "Repo directory ${ITALIAN_SL_REPO} already exists -- pulling latest."
        git -C "$ITALIAN_SL_REPO" pull --quiet
    else
        git clone --quiet \
            https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet \
            "$ITALIAN_SL_REPO"
    fi

    # Measure download size
    download_mb="$(du -sm "$ITALIAN_SL_REPO" 2>/dev/null | cut -f1 || echo "?")"
    ok "Repository cloned (${download_mb} MB at ${ITALIAN_SL_REPO})."

    # Convert to session CSVs
    step "Converting JSON files to session CSVs..."
    mkdir -p "$ITALIAN_SL_RAW"
    python scripts/prepare_italian_sl.py \
        --data-dir "${ITALIAN_SL_REPO}/data/" \
        --output-dir "$ITALIAN_SL_RAW"

    converted="$(find "$ITALIAN_SL_RAW" -name "ISL*.csv" 2>/dev/null | wc -l | tr -d ' ')"
    ok "Converted ${converted} session CSVs into ${ITALIAN_SL_RAW}."
    echo -e "${GREEN}[2/5] Italian SL data: downloaded (${download_mb} MB)${NC}"
fi

# ---------------------------------------------------------------------------
# STEP 3 -- Training
# ---------------------------------------------------------------------------

header "[3/5] Training LSTM on Italian SL data..."

if $SKIP_TRAIN; then
    warn "--skip-train flag set. Skipping training step."

    # Verify a model file actually exists so step 5 won't fail
    if [[ ! -f "${PROJECT_ROOT}/models/asl_emg_classifier.onnx" ]] && \
       [[ ! -f "${PROJECT_ROOT}/models/asl_emg_classifier.pt" ]]; then
        err "No trained model found and --skip-train was set. " \
            "Run without --skip-train at least once to produce a model."
    fi
    echo -e "${GREEN}[3/5] Training: skipped (existing model used)${NC}"
else
    TRAIN_START=$(date +%s)

    # Build argument list for train_real.py
    TRAIN_ARGS=(
        "--data-dir" "$ITALIAN_SL_RAW"
        "--epochs"   "50"
        "--output"   "${PROJECT_ROOT}/models/"
    )

    if $USE_GPU; then
        step "GPU mode requested -- checking CUDA availability..."
        if python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" &>/dev/null 2>&1; then
            ok "CUDA available: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
            # train_real.py auto-detects the device; no extra flag needed
        else
            warn "CUDA not available -- falling back to CPU training."
        fi
    fi

    step "Running: python scripts/train_real.py ${TRAIN_ARGS[*]}"
    # Capture the last line of output to extract accuracy for the status line
    TRAIN_LOG="${PROJECT_ROOT}/logs/pipeline_train_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${PROJECT_ROOT}/logs"
    python scripts/train_real.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"

    TRAIN_END=$(date +%s)
    TRAIN_ELAPSED=$(( TRAIN_END - TRAIN_START ))
    TRAIN_MIN=$(( TRAIN_ELAPSED / 60 ))
    TRAIN_SEC=$(( TRAIN_ELAPSED % 60 ))

    # Try to scrape final validation accuracy from the log
    FINAL_ACC="$(grep -oE 'val_acc[: ]+[0-9]+\.[0-9]+' "$TRAIN_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "?")"
    if [[ "$FINAL_ACC" == "?" ]]; then
        FINAL_ACC="$(grep -oE 'accuracy[: ]+[0-9]+\.[0-9]+' "$TRAIN_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "?")"
    fi

    ok "Training complete. Log saved to: ${TRAIN_LOG}"
    echo -e "${GREEN}[3/5] Training: ${FINAL_ACC}% accuracy (50 epochs, ${TRAIN_MIN}m ${TRAIN_SEC}s)${NC}"
fi

# ---------------------------------------------------------------------------
# STEP 4 -- Smoke tests
# ---------------------------------------------------------------------------

header "[4/5] Running smoke tests..."

SMOKE_LOG="${PROJECT_ROOT}/logs/pipeline_smoke_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${PROJECT_ROOT}/logs"

set +e
python scripts/smoke_test.py 2>&1 | tee "$SMOKE_LOG"
SMOKE_EXIT=${PIPESTATUS[0]}
set -e

# Count PASS / FAIL lines
PASS_COUNT="$(grep -c '^\s*PASS' "$SMOKE_LOG" 2>/dev/null || echo 0)"
FAIL_COUNT="$(grep -c '^\s*FAIL' "$SMOKE_LOG" 2>/dev/null || echo 0)"
TOTAL_COUNT=$(( PASS_COUNT + FAIL_COUNT ))

if [[ $SMOKE_EXIT -ne 0 ]] || [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "${RED}[4/5] Smoke tests: ${PASS_COUNT}/${TOTAL_COUNT} PASS -- ${FAIL_COUNT} FAILED${NC}"
    err "Smoke tests failed. Review ${SMOKE_LOG} for details."
fi

echo -e "${GREEN}[4/5] Smoke tests: ${PASS_COUNT}/${TOTAL_COUNT} PASS${NC}"

# ---------------------------------------------------------------------------
# STEP 5 -- Inference server
# ---------------------------------------------------------------------------

header "[5/5] Starting inference server..."

step "Launching: ./start-server.sh"

# Run the server in the foreground so Ctrl+C stops the script cleanly.
# The start-server.sh script prints its own banner with the URLs.
"${PROJECT_ROOT}/start-server.sh"

# If start-server.sh exits normally (e.g. --prod mode):
echo ""
echo -e "${GREEN}${BOLD}========================================${NC}"
echo -e "${GREEN}${BOLD}  EMG-ASL pipeline complete.${NC}"
echo -e "${GREEN}${BOLD}  REST API:  http://localhost:8000${NC}"
echo -e "${GREEN}${BOLD}  WebSocket: ws://localhost:8000/stream${NC}"
echo -e "${GREEN}${BOLD}  Docs:      http://localhost:8000/docs${NC}"
echo -e "${GREEN}${BOLD}========================================${NC}"
echo ""
echo -e "${GREEN}[5/5] Server: http://localhost:8000 (ws://localhost:8000/stream)${NC}"

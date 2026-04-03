#!/usr/bin/env bash
# EMG-ASL Inference Server Launcher
# Usage: ./start-server.sh [--prod]
# Make executable: chmod +x start-server.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROD_MODE=false
if [[ "${1:-}" == "--prod" ]]; then
    PROD_MODE=true
fi

# --- Activate virtual environment
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}ERROR: No venv found.${NC}"
    echo "  Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo -e "${GREEN}Virtual environment active: $(which python)${NC}"

# --- Ensure baseline model exists
if [ ! -f "models/asl_emg_classifier.onnx" ]; then
    echo -e "${YELLOW}[start-server] No ONNX model found at models/asl_emg_classifier.onnx.${NC}"
    echo -e "${YELLOW}Generating synthetic baseline -- this takes ~60 seconds...${NC}"
    python scripts/train_synthetic_baseline.py
    echo -e "${GREEN}Baseline model generated.${NC}"
else
    echo -e "${GREEN}ONNX model found: models/asl_emg_classifier.onnx${NC}"
fi

# --- Environment
export MAIA_DISABLE_AUTH="${MAIA_DISABLE_AUTH:-1}"
export MODEL_PATH="${MODEL_PATH:-models/asl_emg_classifier.onnx}"
export ONNX_MODEL_PATH="${MODEL_PATH}"

AUTH_STATUS="$([ "$MAIA_DISABLE_AUTH" = "1" ] && echo "DISABLED (dev mode)" || echo "ENABLED")"

echo ""
echo "=============================================="
echo " MAIA EMG-ASL Inference Server"
echo "=============================================="
echo " REST API:  http://localhost:8000"
echo " WebSocket: ws://localhost:8000/stream"
echo " Docs:      http://localhost:8000/docs"
echo " Auth:      ${AUTH_STATUS}"
echo "=============================================="
echo ""

# --- Clear any existing process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "  Cleared stale process on port 8000" || true
echo ""

# --- Launch
if $PROD_MODE; then
    echo -e "${BLUE}Starting PRODUCTION server (4 workers, no reload)...${NC}"
    uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --log-level info \
        2>&1 | tee /tmp/emg-asl-server.log
else
    echo -e "${BLUE}Starting DEVELOPMENT server (auto-reload enabled)...${NC}"
    echo "  Press Ctrl+C to stop"
    echo ""
    uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-dir src \
        --log-level debug \
        2>&1 | tee /tmp/emg-asl-server.log
fi

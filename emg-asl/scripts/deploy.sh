#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — EMG-ASL inference server deployment helper
#
# Usage:
#   ./scripts/deploy.sh           # build + bring stack up
#   ./scripts/deploy.sh --stop    # bring stack down
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Resolve project root (one level up from scripts/) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_NAME="emg-asl-server"
HEALTH_URL="http://localhost:8000/health"
ONNX_MODEL="${PROJECT_ROOT}/models/asl_emg_classifier.onnx"
POLL_INTERVAL=2   # seconds between health-check polls
POLL_MAX=30       # total seconds to wait before giving up

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

die() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC}  $*"
}

ok() {
    echo -e "${GREEN}[OK]${NC}    $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC}  $*"
}

# ─────────────────────────────────────────────────────────────────────────────
# --stop: bring the stack down and exit
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--stop" ]]; then
    info "Stopping EMG-ASL stack..."
    cd "${PROJECT_ROOT}"
    docker compose down
    ok "Stack stopped."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Check Docker is installed and the daemon is reachable
# ─────────────────────────────────────────────────────────────────────────────

if ! command -v docker &>/dev/null; then
    die "Docker is not installed. Install from https://docs.docker.com/get-docker/"
fi

if ! docker info &>/dev/null; then
    die "Docker daemon is not running. Start Docker Desktop (or 'sudo systemctl start docker') and retry."
fi

ok "Docker is available: $(docker --version)"

# Check docker compose plugin (v2) or standalone docker-compose (v1)
if docker compose version &>/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
else
    die "Neither 'docker compose' (v2 plugin) nor 'docker-compose' (v1) found."
fi

ok "Using compose command: ${COMPOSE_CMD}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Warn if no trained model exists (server will start in random-weight mode)
# ─────────────────────────────────────────────────────────────────────────────

if [[ -f "${ONNX_MODEL}" ]]; then
    ok "ONNX model found: ${ONNX_MODEL}"
else
    warn "No trained model found at:"
    warn "  ${ONNX_MODEL}"
    warn "The server will start but predictions will return random weights."
    warn "To train a model run:"
    warn "  ${COMPOSE_CMD} --profile training up model-trainer"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the Docker image
# ─────────────────────────────────────────────────────────────────────────────

cd "${PROJECT_ROOT}"

info "Building image '${IMAGE_NAME}' from ${PROJECT_ROOT}..."
docker build -t "${IMAGE_NAME}" .
ok "Image built successfully."

# ─────────────────────────────────────────────────────────────────────────────
# 4. Bring the stack up (detached)
# ─────────────────────────────────────────────────────────────────────────────

info "Starting inference stack (detached)..."
${COMPOSE_CMD} up -d
ok "Containers started."

# ─────────────────────────────────────────────────────────────────────────────
# 5. Wait for /health to return 200
# ─────────────────────────────────────────────────────────────────────────────

info "Waiting for server to become healthy (max ${POLL_MAX}s)..."

elapsed=0
healthy=false

while (( elapsed < POLL_MAX )); do
    if curl -sf "${HEALTH_URL}" -o /dev/null 2>/dev/null; then
        healthy=true
        break
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$(( elapsed + POLL_INTERVAL ))
    echo -n "."
done

echo ""  # newline after dots

if [[ "${healthy}" == "true" ]]; then
    ok "Server is healthy (responded in ~${elapsed}s)."
else
    warn "Server did not respond at ${HEALTH_URL} within ${POLL_MAX}s."
    warn "Check container logs with: ${COMPOSE_CMD} logs inference-server"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Print access URLs
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}EMG-ASL inference server is running${NC}"
echo ""
echo "  REST API:  http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  WebSocket: ws://localhost:8765/stream"
echo ""
echo "  Logs:      ${COMPOSE_CMD} logs -f inference-server"
echo "  Stop:      $0 --stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

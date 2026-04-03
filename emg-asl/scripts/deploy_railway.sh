#!/usr/bin/env bash
# deploy_railway.sh -- Deploy MAIA EMG-ASL inference server to Railway
#
# Prerequisites:
#   npm install -g @railway/cli   (or: brew install railway)
#   railway login
#
# First deploy:
#   ./scripts/deploy_railway.sh --setup
#
# Subsequent deploys:
#   ./scripts/deploy_railway.sh
#
# Usage:
#   ./scripts/deploy_railway.sh [--setup] [--env ENV_FILE]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SETUP=false
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup)
            SETUP=true
            shift
            ;;
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            error "Unknown argument: $1\nUsage: $0 [--setup] [--env ENV_FILE]"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Check railway CLI is installed
# ---------------------------------------------------------------------------

if ! command -v railway &>/dev/null; then
    error "railway CLI not found.\n\nInstall it with one of:\n  npm install -g @railway/cli\n  brew install railway\n\nThen run: railway login"
fi

info "Found railway CLI: $(railway --version 2>/dev/null || echo '(version unknown)')"

# ---------------------------------------------------------------------------
# Check the user is logged in
# ---------------------------------------------------------------------------

if ! railway whoami &>/dev/null; then
    error "Not logged in to Railway. Run: railway login"
fi

RAILWAY_USER="$(railway whoami 2>/dev/null)"
info "Logged in as: ${RAILWAY_USER}"

# ---------------------------------------------------------------------------
# Setup mode: initialise project and set environment variables
# ---------------------------------------------------------------------------

if [[ "$SETUP" == "true" ]]; then
    info "Running first-time setup..."

    info "Initialising Railway project (follow the prompts)..."
    railway init

    info "Setting required environment variables..."
    railway variables set MAIA_DISABLE_AUTH=0
    railway variables set MAIA_API_KEYS=your-api-key-here
    railway variables set ONNX_MODEL_PATH=models/asl_emg_classifier.onnx
    railway variables set R2_MODEL_URL=""

    echo ""
    echo -e "${BOLD}Setup complete. Next steps:${NC}"
    echo ""
    echo "  1. Upload your ONNX model to Cloudflare R2:"
    echo "       python scripts/upload_artifact.py models/asl_emg_classifier.onnx"
    echo ""
    echo "  2. Set the R2 public URL in Railway:"
    echo "       railway variables set R2_MODEL_URL=https://pub-xxxx.r2.dev/models/asl_emg_classifier.onnx"
    echo ""
    echo "  3. Set a real API key (generate one with):"
    echo "       python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    echo "       railway variables set MAIA_API_KEYS=<generated-key>"
    echo ""
    echo "  4. Deploy:"
    echo "       ./scripts/deploy_railway.sh"
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Optional: load extra env vars from a file before deploying
# ---------------------------------------------------------------------------

if [[ -n "$ENV_FILE" ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Env file not found: $ENV_FILE"
    fi
    info "Loading environment variables from: $ENV_FILE"
    while IFS='=' read -r key value; do
        # Skip comments and blank lines
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        railway variables set "${key}=${value}"
        info "  Set: $key"
    done < "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

info "Deploying to Railway (detached)..."
railway up --detach

# ---------------------------------------------------------------------------
# Post-deploy: print status and test the health endpoint
# ---------------------------------------------------------------------------

info "Fetching deployment status..."
railway status

# Extract the public domain from `railway domain`
RAILWAY_DOMAIN="$(railway domain 2>/dev/null | grep -Eo '[a-zA-Z0-9.-]+\.railway\.app' | head -1 || true)"

if [[ -n "$RAILWAY_DOMAIN" ]]; then
    BASE_URL="https://${RAILWAY_DOMAIN}"
    echo ""
    echo -e "${BOLD}Deployed URL:${NC}        ${BASE_URL}"
    echo -e "${BOLD}Health endpoint:${NC}     ${BASE_URL}/health"
    echo -e "${BOLD}Info endpoint:${NC}       ${BASE_URL}/info"
    echo -e "${BOLD}WebSocket endpoint:${NC}  wss://${RAILWAY_DOMAIN}/stream"
    echo ""

    info "Testing /health endpoint (may take a few seconds for the container to boot)..."
    # Retry up to 10 times with a 5-second gap
    for attempt in $(seq 1 10); do
        HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" || true)"
        if [[ "$HTTP_CODE" == "200" ]]; then
            HEALTH_BODY="$(curl -s "${BASE_URL}/health")"
            info "Health check passed (attempt ${attempt}): ${HEALTH_BODY}"
            break
        else
            warn "Attempt ${attempt}/10: /health returned HTTP ${HTTP_CODE}. Retrying in 5s..."
            sleep 5
        fi
    done

    if [[ "$HTTP_CODE" != "200" ]]; then
        warn "Health check did not return 200 after 10 attempts. The service may still be starting."
        warn "Check logs with: railway logs"
    fi
else
    warn "Could not determine Railway domain automatically."
    warn "Run 'railway domain' to get the public URL."
fi

info "Done."

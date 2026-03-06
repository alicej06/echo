#!/usr/bin/env bash
# =============================================================================
# download_datasets.sh — one-command dataset downloader for EMG-ASL layer
#
# Usage:
#   bash scripts/download_datasets.sh              # print menu and exit
#   bash scripts/download_datasets.sh --grabmyo    # download GRABMyo (~9.4 GB)
#   bash scripts/download_datasets.sh --italian-sl # clone Italian SL dataset
#   bash scripts/download_datasets.sh --all        # run all auto-downloadable
#
# Must be run from the project root (the directory that contains src/).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers (gracefully degrade when stdout is not a terminal)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD="\033[1m"
    GREEN="\033[0;32m"
    YELLOW="\033[0;33m"
    CYAN="\033[0;36m"
    RED="\033[0;31m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" CYAN="" RED="" RESET=""
fi

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# Ensure we are at the project root (basic sanity check)
# ---------------------------------------------------------------------------
if [ ! -d "src" ] || [ ! -f "requirements.txt" ]; then
    error "Run this script from the project root directory (the one containing src/)."
    exit 1
fi

# ---------------------------------------------------------------------------
# Target directories
# ---------------------------------------------------------------------------
GRABMYO_DIR="data/external/grabmyo"
ITALIAN_SL_DIR="data/external/italian-sl"

# ---------------------------------------------------------------------------
# print_menu — show what is available and how to download it
# ---------------------------------------------------------------------------
print_menu() {
    echo ""
    echo -e "${BOLD}EMG-ASL Dataset Downloader${RESET}"
    echo "=============================="
    echo ""
    echo -e "${BOLD}Auto-downloadable datasets:${RESET}"
    echo ""
    echo -e "  ${GREEN}--grabmyo${RESET}"
    echo "      GRABMyo — Grasp-and-Release with Bio-signals (PhysioNet)"
    echo "      Format  : WFDB (read with wfdb>=4.1)"
    echo "      Size    : ~9.4 GB"
    echo "      Target  : ${GRABMYO_DIR}/"
    echo "      Command : bash scripts/download_datasets.sh --grabmyo"
    echo ""
    echo -e "  ${GREEN}--italian-sl${RESET}"
    echo "      Italian Sign Language Alphabet — EMG + IMU (GitHub / airtlab)"
    echo "      Format  : CSV, ~120 MB"
    echo "      Target  : ${ITALIAN_SL_DIR}/"
    echo "      Command : bash scripts/download_datasets.sh --italian-sl"
    echo ""
    echo -e "  ${GREEN}--all${RESET}"
    echo "      Run both --grabmyo and --italian-sl sequentially."
    echo ""
    echo "----------------------------------------------------------------------"
    echo -e "${BOLD}Manual-download datasets (no public direct URL):${RESET}"
    echo ""
    echo -e "  ${YELLOW}ASLA (RIT — American Sign Language Archive)${RESET}"
    echo "      Request access : https://www.rit.edu/ntid/slac/resources"
    echo "      Place files in : data/external/asla-rit/"
    echo ""
    echo -e "  ${YELLOW}NinaProDB (DB1–DB8)${RESET}"
    echo "      Register & download : http://ninapro.hevs.ch"
    echo "      Prepare with        : python scripts/prepare_ninapro.py"
    echo "      Place files in      : data/external/ninapro/"
    echo ""
    echo -e "  ${YELLOW}Mendeley ASL sEMG Dataset${RESET}"
    echo "      DOI    : https://doi.org/10.17632/ckwc76xr2z.1"
    echo "      Log in to Mendeley Data and click Download All."
    echo "      Place files in : data/external/mendeley-asl/"
    echo ""
}

# ---------------------------------------------------------------------------
# download_grabmyo
# ---------------------------------------------------------------------------
download_grabmyo() {
    echo ""
    info "GRABMyo dataset via PhysioNet"
    echo ""
    warn "This download is approximately 9.4 GB."
    warn "Destination: ${GRABMYO_DIR}/"
    echo ""
    read -r -p "Continue? [y/N] " confirm
    echo ""

    if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
        warn "GRABMyo download cancelled."
        return 0
    fi

    if ! command -v wget &>/dev/null; then
        error "wget is required but not found. Install it (brew install wget) and retry."
        return 1
    fi

    mkdir -p "${GRABMYO_DIR}"
    info "Starting wget mirror of physionet.org/files/grabmyo/1.1.0/ …"
    echo "  (flags: -r recursive, -N timestamping, -c continue, -np no-parent)"
    echo ""

    wget \
        --recursive \
        --timestamping \
        --continue \
        --no-parent \
        --directory-prefix="${GRABMYO_DIR}" \
        https://physionet.org/files/grabmyo/1.1.0/

    success "GRABMyo download complete → ${GRABMYO_DIR}/"
    info "Prepare with: python scripts/prepare_grabmyo.py"
}

# ---------------------------------------------------------------------------
# download_italian_sl
# ---------------------------------------------------------------------------
download_italian_sl() {
    echo ""
    info "Italian Sign Language Alphabet dataset (airtlab / GitHub)"

    if ! command -v git &>/dev/null; then
        error "git is required but not found. Install Xcode Command Line Tools and retry."
        return 1
    fi

    if [ -d "${ITALIAN_SL_DIR}/.git" ]; then
        info "Repository already cloned at ${ITALIAN_SL_DIR}/ — pulling latest."
        git -C "${ITALIAN_SL_DIR}" pull
    else
        mkdir -p "$(dirname "${ITALIAN_SL_DIR}")"
        info "Cloning into ${ITALIAN_SL_DIR}/ …"
        git clone \
            https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet \
            "${ITALIAN_SL_DIR}"
    fi

    success "Italian SL dataset ready → ${ITALIAN_SL_DIR}/"
}

# ---------------------------------------------------------------------------
# Argument dispatch
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    print_menu
    exit 0
fi

RUN_GRABMYO=false
RUN_ITALIAN_SL=false

for arg in "$@"; do
    case "${arg}" in
        --grabmyo)
            RUN_GRABMYO=true
            ;;
        --italian-sl)
            RUN_ITALIAN_SL=true
            ;;
        --all)
            RUN_GRABMYO=true
            RUN_ITALIAN_SL=true
            ;;
        --help|-h)
            print_menu
            exit 0
            ;;
        *)
            error "Unknown argument: ${arg}"
            echo "Usage: bash scripts/download_datasets.sh [--grabmyo] [--italian-sl] [--all]"
            exit 1
            ;;
    esac
done

if "${RUN_GRABMYO}";    then download_grabmyo;    fi
if "${RUN_ITALIAN_SL}"; then download_italian_sl; fi

echo ""
success "All requested downloads finished."

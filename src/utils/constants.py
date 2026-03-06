"""
EMG-ASL inference server constants.

Covers ASL label vocabulary, signal processing parameters,
server networking settings, and inference tuning knobs.

Hardware: Thalmic MYO Armband (8-channel sEMG, 200 Hz).
Connection is managed via myo-python + MyoConnect (USB dongle required).
See hardware/myo_armband/README.md for setup instructions.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# ASL label vocabulary
# 26 letters A-Z followed by 10 common-word tokens.
# The model output index maps 1-to-1 with this list.
# ---------------------------------------------------------------------------

ASL_LETTERS: list[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

ASL_WORDS: list[str] = [
    "HELLO",
    "THANK_YOU",
    "PLEASE",
    "YES",
    "NO",
    "HELP",
    "WATER",
    "MORE",
    "STOP",
    "GO",
]

ASL_LABELS: list[str] = ASL_LETTERS + ASL_WORDS
NUM_CLASSES: int = len(ASL_LABELS)  # 36

# ---------------------------------------------------------------------------
# EMG signal-processing constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 200          # Hz — samples per second per channel
WINDOW_SIZE_MS: int = 200       # milliseconds per analysis window
OVERLAP: float = 0.5            # fraction of window overlap between successive windows
N_CHANNELS: int = 8             # number of EMG electrodes / BLE channels

# Derived window dimensions
WINDOW_SIZE_SAMPLES: int = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)   # 40 samples
STEP_SIZE_SAMPLES: int = int(WINDOW_SIZE_SAMPLES * (1.0 - OVERLAP))   # 20 samples

# Butterworth bandpass filter corners (Hz)
BANDPASS_LOW: float = 20.0
BANDPASS_HIGH: float = 450.0

# IIR notch filter to reject power-line interference (Hz)
NOTCH_FREQ: float = 60.0

# ---------------------------------------------------------------------------
# Server / networking constants
# ---------------------------------------------------------------------------

WS_HOST: str = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT: int = int(os.environ.get("WS_PORT", "8765"))
REST_PORT: int = int(os.environ.get("REST_PORT", "8000"))

# ---------------------------------------------------------------------------
# Inference quality / UX constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.75   # predictions below this are suppressed
DEBOUNCE_MS: int = 300               # minimum gap (ms) between consecutive label emissions

# ---------------------------------------------------------------------------
# Model & calibration paths (overridable via environment)
# ---------------------------------------------------------------------------

MODEL_DIR: str = os.environ.get("MODEL_DIR", "models")
ONNX_MODEL_PATH: str = os.environ.get(
    "ONNX_MODEL_PATH", f"{MODEL_DIR}/asl_emg_classifier.onnx"
)
PYTORCH_MODEL_PATH: str = os.environ.get(
    "PYTORCH_MODEL_PATH", f"{MODEL_DIR}/asl_emg_classifier.pt"
)
PROFILE_DIR: str = os.environ.get("PROFILE_DIR", "profiles")

# ---------------------------------------------------------------------------
# MYO Armband — device name and BLE UUIDs
#
# Primary connection method: myo-python SDK (via MyoConnect + USB dongle).
# myo-python abstracts the BLE layer; these UUIDs are only needed for
# direct BLE access from the mobile app (bypassing MyoConnect).
#
# MYO direct-BLE UUIDs are public (reverse-engineered from the Thalmic SDK):
#   mobile app: mobile/react-native/src/bluetooth/BLEManager.ts
# ---------------------------------------------------------------------------

BLE_DEVICE_NAME: str = "Myo"

# MYO EMG service (4 notify characteristics, each streams 2 samples × 4 ch at 50 Hz)
MYO_EMG_SERVICE_UUID: str = "d5060005-a904-deb9-4748-2c7f4a124842"
MYO_EMG_CHAR_UUIDS: list[str] = [
    "d5060105-a904-deb9-4748-2c7f4a124842",  # channels 0-3, samples 0 & 1
    "d5060205-a904-deb9-4748-2c7f4a124842",  # channels 4-7, samples 0 & 1
    "d5060305-a904-deb9-4748-2c7f4a124842",  # channels 0-3, samples 2 & 3
    "d5060405-a904-deb9-4748-2c7f4a124842",  # channels 4-7, samples 2 & 3
]

# Backwards-compat aliases used by recorder.py / record_session.py
BLE_EMG_SERVICE_UUID: str = MYO_EMG_SERVICE_UUID
BLE_EMG_CHAR_UUID: str = MYO_EMG_CHAR_UUIDS[0]  # primary char (fallback)

# ---------------------------------------------------------------------------
# Feature extraction — derived sizes used for model input validation
# ---------------------------------------------------------------------------

# Time-domain features per channel: RMS, MAV, WL, ZC, SSC  -> 5 features
N_TIME_FEATURES_PER_CHANNEL: int = 5
# Frequency-domain features per channel: mean_freq, median_freq, moment2, moment3, moment4 -> 5 features
N_FREQ_FEATURES_PER_CHANNEL: int = 5
N_FEATURES_PER_CHANNEL: int = N_TIME_FEATURES_PER_CHANNEL + N_FREQ_FEATURES_PER_CHANNEL
FEATURE_VECTOR_SIZE: int = N_FEATURES_PER_CHANNEL * N_CHANNELS  # 80 features total

"""MAIA EMG-ASL signal and model constants.

Hardware target: Thalmic Myo Armband (8 channels, 200Hz, int8 output)
Connection: direct BLE GATT protocol on iOS (no dongle required).

Research datasets for pre-training: facebookresearch/emg2pose,
facebookresearch/generic-neuromotor-interface, GRABMyo, NinaPro DB2-9.
"""

# --- Signal acquisition (Myo Armband spec) ---
SAMPLE_RATE    = 200          # Hz  (200 Hz per channel, 8 channels)
N_CHANNELS     = 8            # sEMG channels (Myo Armband)
WINDOW_SAMPLES = 40           # 200ms @ 200Hz
HOP_SAMPLES    = 20           # 50% overlap → 100ms hop
WINDOW_MS      = 200          # ms
HOP_MS         = 100          # ms

# --- Feature extraction ---
N_FEATURES_PER_CHANNEL = 10
FEATURE_DIM = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 80

# --- Classes ---
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
N_CLASSES = len(ASL_CLASSES)  # 26

# --- BLE packet format (Myo Armband direct-BLE) ---
# Each EMG notification = 16 bytes: 2 samples × 8 channels × int8
BLE_NOTIFICATION_BYTES = 16
BLE_SAMPLES_PER_NOTIF  = 2
BLE_BYTES_PER_SAMPLE   = N_CHANNELS  # 8 int8 values

# Myo BLE UUIDs (thalmiclabs/myo-bluetooth-protocol)
# Note: service UUID ends in 0001, NOT 0005 (0005 is the classifier service)
BLE_SERVICE_UUID    = "d5060001-a904-deb9-4748-2c7f4a124842"
BLE_EMG_CHAR_UUIDS  = [
    "d5060105-a904-deb9-4748-2c7f4a124842",  # EMG data 0
    "d5060205-a904-deb9-4748-2c7f4a124842",  # EMG data 1
    "d5060305-a904-deb9-4748-2c7f4a124842",  # EMG data 2
    "d5060405-a904-deb9-4748-2c7f4a124842",  # EMG data 3
]
BLE_CONTROL_UUID    = "d5060401-a904-deb9-4748-2c7f4a124842"

# Command to enable raw EMG streaming (set_mode: emg=raw, imu=off, classifier=off)
MYO_ENABLE_EMG_CMD = bytes([0x01, 0x03, 0x02, 0x00, 0x00])

# --- WebSocket frame ---
# Floats from EMGWindowBuffer are converted to int16 before sending to Railway WS
WS_FRAME_BYTES = WINDOW_SAMPLES * N_CHANNELS * 2  # 640 bytes (int16)

# --- Model defaults ---
LSTM_HIDDEN       = 128       # 8ch input — 128 hidden is sufficient
LSTM_LAYERS       = 2
LSTM_DROPOUT      = 0.3
CNN_FILTERS       = [32, 64]
CONFORMER_D_MODEL = 128
CONFORMER_N_HEADS = 4
CONFORMER_N_LAYERS = 4

# --- Training ---
BATCH_SIZE          = 256
LEARNING_RATE       = 1e-3
MAX_EPOCHS          = 200
EARLY_STOP_PATIENCE = 15

# --- Inference ---
CONFIDENCE_THRESHOLD = 0.75
DEBOUNCE_MS          = 300

# --- Third-party research dataset specs (for loaders) ---
# Meta/NinaPro datasets are 16ch/2kHz — resampled to 200Hz and subset to 8ch
META_SAMPLE_RATE    = 2000    # Hz — source sample rate; resampled to SAMPLE_RATE
META_N_CHANNELS     = 16      # source channel count; subset to N_CHANNELS
META_HIGHPASS_HZ    = 40.0    # Meta pre-applied high-pass filter cutoff
META_TASKS          = ("discrete_gestures", "handwriting", "wrist")
NINAPRO_SAMPLE_RATE = 2000    # Hz (DB5+) — resampled to SAMPLE_RATE

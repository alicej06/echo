"""MAIA EMG-ASL signal and model constants.

Hardware target: Thalmic MYO Armband (8 channels, 200 Hz, int8 output)
Connection: myo-python + MyoConnect (USB dongle) on laptop/server;
            direct BLE GATT protocol on iOS (no dongle required).

Research datasets used for pre-training: facebookresearch/emg2pose,
facebookresearch/generic-neuromotor-interface, GRABMyo, NinaPro DB2-9.
"""

# --- Signal acquisition (MYO Armband spec) ---
SAMPLE_RATE    = 200          # Hz  (200 Hz per channel, 8 channels)
N_CHANNELS     = 8            # sEMG channels (MYO Armband)
WINDOW_SAMPLES = 40           # 200ms @ 200 Hz
HOP_SAMPLES    = 20           # 50% overlap → 100ms hop
WINDOW_MS      = 200          # ms
HOP_MS         = 100          # ms

# --- Feature extraction ---
N_FEATURES_PER_CHANNEL = 10
FEATURE_DIM = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 160

# --- Classes ---
ASL_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
N_CLASSES = len(ASL_CLASSES)  # 26

# --- BLE packet format (MYO Armband direct-BLE) ---
BLE_PACKET_BYTES  = 8    # 2 samples × 4 channels × int8 = 8 bytes per EMG char
# MYO Armband GATT UUIDs (reverse-engineered from Thalmic SDK)
BLE_SERVICE_UUID  = "d5060005-a904-deb9-4748-2c7f4a124842"
BLE_CHAR_UUID     = "d5060105-a904-deb9-4748-2c7f4a124842"  # primary EMG char

# --- WebSocket frame ---
WS_FRAME_BYTES = WINDOW_SAMPLES * N_CHANNELS * 1  # 320 bytes (int8)

# --- Model defaults ---
LSTM_HIDDEN       = 128       # 8ch input is small — 128 hidden is sufficient
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

# --- Meta/third-party research dataset specs (for loaders) ---
# Meta datasets are 16ch/2kHz — resampled to 200 Hz and channel-subset to 8ch
META_SAMPLE_RATE    = 2000    # Hz — source sample rate; resampled to SAMPLE_RATE
META_N_CHANNELS     = 16      # source channel count; subset to N_CHANNELS
META_HIGHPASS_HZ    = 40.0    # Meta pre-applied high-pass filter cutoff
META_TASKS          = ("discrete_gestures", "handwriting", "wrist")
NINAPRO_SAMPLE_RATE = 2000    # Hz (DB5+) — resampled to SAMPLE_RATE

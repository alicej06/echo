# EMG-ASL Layer — Technical Architecture

## Table of Contents

1. [System Overview](#1-system-overview)
2. [BLE Data Format Specification](#2-ble-data-format-specification)
3. [Signal Processing Pipeline](#3-signal-processing-pipeline)
4. [ML Model Architecture](#4-ml-model-architecture)
5. [API Specification](#5-api-specification)
6. [Mobile App Component Diagram](#6-mobile-app-component-diagram)
7. [Latency Budget Analysis](#7-latency-budget-analysis)
8. [Security Considerations](#8-security-considerations)

---

## 1. System Overview

The EMG-ASL Layer is a four-tier real-time system:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EMG-ASL Full System Diagram                           │
└─────────────────────────────────────────────────────────────────────────────────┘

 SENSOR LAYER                  MOBILE LAYER              INFERENCE LAYER
 ┌────────────────────┐       ┌────────────────────┐    ┌──────────────────────────┐
 │  Thalmic MYO       │       │  React Native App  │    │  Python Inference Server │
 │  Armband           │       │                    │    │                          │
 │  8-ch sEMG, 200 Hz │       │  BLEManager.ts     │    │  ws_handler.py           │
 │  int8, ±1.25 mV    │       │  (direct MYO BLE)  │    │   ↓ raw bytes            │
 └────────┬───────────┘       │   ↓ 4 EMG chars    │    │  preprocessor.py         │
          │ MYO BLE            │  WebSocketClient   │    │   ↓ filtered signal      │
          │ 4 notify chars     │   ↓ JSON/binary    │    │  feature_extractor.py    │
          │ 8 ch @ 200 Hz      └────────┬───────────┘    │   ↓ 80-dim vector        │
          │                            │  WebSocket      │  inference.py (ONNX)     │
          ▼ (laptop path)              │  ws://LAN:8765  │   ↓ label + confidence   │
 ┌────────────────────┐               │                 │  debounce logic          │
 │  MYO USB Dongle    │               │                 └──────────────────────────┘
 │  + MyoConnect      │               │                             │
 │  + myo-python      │               │                             │
 │  record_session.py │               │                             │
 └────────────────────┘      ┌────────▼───────────┐                │
                             │  React Native App  │◄───────────────┘ JSON response
                             │  (response side)   │
                             │  TranscriptView.tsx│
                             │  expo-speech TTS   │
                             └────────────────────┘

 DATA STORES
 ┌────────────────────────────────────────────────────────────┐
 │  data/raw/          ← session CSVs (gitignored)            │
 │  data/processed/    ← windowed feature tensors (gitignored)│
 │  models/            ← ONNX + PyTorch weights (gitignored)  │
 │  profiles/          ← per-user calibration JSON            │
 └────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **MYO Armband**: Pre-built 8-channel sEMG device (200 Hz, int8). No custom firmware or wiring required. Connects via myo-python + MyoConnect (laptop) or direct BLE GATT protocol (mobile). Eliminates the nRF52840 + MyoWare custom hardware path entirely.
- **Server-side inference**: Keeps the MCU simple (no ML on device). CPU-only ONNX inference is fast enough (~15 ms) for real-time use.
- **ONNX export**: Decouples training framework (PyTorch) from runtime, enabling server portability without GPU drivers.
- **Per-user calibration**: Fine-tunes a global model's final classification layer with ~50 labeled windows per class from the specific user, improving cross-session accuracy from ~82% to ~94%.

---

## 2. BLE Data Format Specification

### MYO BLE Service and Characteristic UUIDs

The authoritative source is `src/utils/constants.py`. UUIDs are from the
publicly reverse-engineered MYO BLE GATT profile.

```
Control Service:         d5060001-a904-deb9-4748-2c7f4a124842
Command Characteristic:  d5060401-a904-deb9-4748-2c7f4a124842

EMG Service:             d5060005-a904-deb9-4748-2c7f4a124842
EMG Characteristic 0:    d5060105-a904-deb9-4748-2c7f4a124842  (ch 0–3, samples 0–1)
EMG Characteristic 1:    d5060205-a904-deb9-4748-2c7f4a124842  (ch 4–7, samples 0–1)
EMG Characteristic 2:    d5060305-a904-deb9-4748-2c7f4a124842  (ch 0–3, samples 2–3)
EMG Characteristic 3:    d5060405-a904-deb9-4748-2c7f4a124842  (ch 4–7, samples 2–3)
```

### MYO BLE Notification Format

Each of the 4 EMG notify characteristics delivers 2 samples × 4 channels (int8)
at ~50 Hz. All 4 characteristics together reconstruct 8 channels at 200 Hz.

```
 Notification (8 bytes):
  [s0_ch0, s0_ch1, s0_ch2, s0_ch3, s1_ch0, s1_ch1, s1_ch2, s1_ch3]
  Type: int8 (signed), range −128 to 127

 To millivolts: mv = value / 127.0 * 1250.0  →  ±1.25 mV range
```

### SET_EMG_MODE Command

Before subscribing to EMG characteristics, write this 5-byte command to the
Command Characteristic to enable raw filtered 200 Hz streaming:

```
[0x01, 0x03, 0x02, 0x01, 0x01]
 cmd   size  emg   imu   classifier
```

### MYO Advertising

- Device name: `Myo` (prefix-matches; may also appear as `Myo (XXXX)`)
- After connection: discover services → write SET_EMG_MODE → subscribe to all 4 EMG chars
- The app (BLEManager.ts) handles reconnection with exponential backoff (max 5 attempts)

---

## 3. Signal Processing Pipeline

All processing runs in `src/data/preprocessor.py` and `src/data/feature_extractor.py`.

### 3.1 Filtering

**Step 1 — Bandpass Filter (20–450 Hz)**

Removes DC offset and high-frequency noise. Parameters:

```
Filter type:  Butterworth IIR, order 4
Passband:     20 Hz – 450 Hz
Sample rate:  200 Hz
              Note: 450 Hz exceeds Nyquist (100 Hz) so effective highcut
              is clamped to 95 Hz in implementation to avoid instability.
              Set BANDPASS_HIGH = 95.0 in practice; the constant is named
              for the biological target frequency range.
Implemented:  scipy.signal.butter + sosfiltfilt (zero-phase, forward-backward)
```

**Step 2 — Notch Filter (60 Hz)**

Removes AC power-line interference:

```
Filter type:  IIR notch (second-order IIR)
Center freq:  60 Hz (configurable; 50 Hz for EU hardware)
Q factor:     30.0
Implemented:  scipy.signal.iirnotch + sosfiltfilt
```

### 3.2 Windowing

Sliding window applied after filtering:

```
Window size:  200 ms = 40 samples (at 200 Hz)
Overlap:      50%  → step size = 20 samples (100 ms step)
Shape:        window_tensor.shape = (40, 8)   [samples × channels]
```

### 3.3 Feature Extraction

From each 40×8 window, 10 features are extracted per channel = 80 features total.

**Time-Domain Features (5 per channel)**

| Feature | Symbol | Formula |
|---------|--------|---------|
| Root Mean Square | RMS | `sqrt(mean(x²))` |
| Mean Absolute Value | MAV | `mean(|x|)` |
| Waveform Length | WL | `sum(|x[i] - x[i-1]|)` |
| Zero Crossings | ZC | `sum(sign(x[i]) != sign(x[i-1]))` with threshold 0.01 mV |
| Slope Sign Changes | SSC | `sum(sign(x[i]-x[i-1]) != sign(x[i+1]-x[i]))` |

**Frequency-Domain Features (5 per channel)**

Computed via `numpy.fft.rfft` on the windowed channel signal:

| Feature | Symbol | Formula |
|---------|--------|---------|
| Mean Frequency | MNF | `sum(f * PSD(f)) / sum(PSD(f))` |
| Median Frequency | MDF | freq s.t. `cumsum(PSD) >= 0.5 * sum(PSD)` |
| Second Spectral Moment | SM2 | `sum((f - MNF)² * PSD(f)) / sum(PSD(f))` |
| Third Spectral Moment | SM3 | `sum((f - MNF)³ * PSD(f)) / sum(PSD(f))` |
| Fourth Spectral Moment | SM4 | `sum((f - MNF)⁴ * PSD(f)) / sum(PSD(f))` |

**Final Feature Vector**

```
Shape: (80,)  = 8 channels × 10 features/channel
Order: [ch0_RMS, ch0_MAV, ch0_WL, ch0_ZC, ch0_SSC, ch0_MNF, ch0_MDF, ch0_SM2, ch0_SM3, ch0_SM4,
        ch1_RMS, ..., ch7_SM4]
```

### 3.4 Normalization

Feature vectors are z-score normalized using per-feature statistics (`mean`, `std`) computed on the training set and stored alongside the model in `models/scaler.json`. At inference time the server loads this JSON and applies `(x - mean) / std` before passing to the model.

---

## 4. ML Model Architecture

### 4.1 Model: Temporal Convolutional + LSTM Classifier

```
Input:  feature sequence of shape (T, 80)
        where T = number of sliding windows in one sign gesture (typically 5–15)

Layer 1:  Linear(80 → 128) + ReLU + Dropout(0.3)
Layer 2:  Linear(128 → 128) + ReLU + Dropout(0.3)
Layer 3:  LSTM(input=128, hidden=256, num_layers=2, batch_first=True, dropout=0.3)
Layer 4:  Take last hidden state → Linear(256 → 128) + ReLU
Layer 5:  Linear(128 → 36)   [36 = 26 letters + 10 words]
Output:   Softmax probabilities over 36 classes
```

For real-time inference (streaming mode), the server uses a **single-window** variant where T=1 and the LSTM state is carried across windows using a stateful inference session. Each new window updates the hidden state and produces an updated class probability vector.

### 4.2 Training Procedure

```
Framework:      PyTorch 2.x
Optimizer:      AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler:      CosineAnnealingLR (T_max=100 epochs, eta_min=1e-5)
Loss:           CrossEntropyLoss with label smoothing=0.1
Batch size:     32
Max epochs:     100
Early stopping: patience=15 (monitors validation loss)
Train/val/test: 70/15/15 stratified split (per-participant held-out test)
Augmentation:   Gaussian noise (σ=0.01), random amplitude scaling (0.8–1.2×)
```

### 4.3 Per-User Calibration (Fine-Tuning)

The global model's final linear layer (Linear(128 → 36)) is re-trained per user using a small labeled set collected via the calibration flow in the mobile app. The body of the network is frozen.

```
Calibration data:  ~50 windows per class (collected in ~5 minutes)
Fine-tune layers:  Layer 4 (Linear 256→128) and Layer 5 (Linear 128→36)
Optimizer:         Adam (lr=5e-4)
Epochs:            30 (no early stopping; small data)
```

### 4.4 ONNX Export

```python
torch.onnx.export(
    model,
    dummy_input,                  # shape (1, 1, 80) — batch=1, seq_len=1, features=80
    "models/asl_emg_classifier.onnx",
    input_names=["feature_vector"],
    output_names=["class_logits"],
    dynamic_axes={"feature_vector": {0: "batch_size", 1: "seq_len"}},
    opset_version=17,
)
```

---

## 5. API Specification

### 5.1 WebSocket — `/stream`

**Endpoint**: `ws://<server>:8765/stream`

**Connection procedure**:
1. Client opens WebSocket connection.
2. Server sends handshake JSON: `{"type": "connected", "server_version": "1.0.0", "num_classes": 36}`.
3. Client sends binary frames continuously at ~200 Hz (each frame = 16 bytes, one EMG sample).
4. Server accumulates samples into a sliding window buffer and sends a JSON response each time a new window is complete.
5. On disconnect, server cleans up LSTM hidden state for that connection.

**Binary Frame (Client → Server)**:
```
16 bytes: ch0_hi, ch0_lo, ch1_hi, ch1_lo, ..., ch7_hi, ch7_lo
          (8 × int16, big-endian)
```

**JSON Response (Server → Client)**:
```json
{
  "type": "prediction",
  "label": "HELLO",
  "label_index": 26,
  "confidence": 0.94,
  "suppressed": false,
  "timestamp_ms": 1740000000123
}
```

If `confidence < CONFIDENCE_THRESHOLD` (0.75) or debounce prevents emission:
```json
{
  "type": "prediction",
  "label": "B",
  "label_index": 1,
  "confidence": 0.61,
  "suppressed": true,
  "timestamp_ms": 1740000000456
}
```

**Error Codes** (sent as JSON before close):
```json
{"type": "error", "code": 4001, "message": "Invalid frame length: expected 16 bytes, got 12"}
{"type": "error", "code": 4002, "message": "Model not loaded. Server is not ready."}
{"type": "error", "code": 4003, "message": "Too many connections. Server at capacity (max 4)."}
```

---

### 5.2 REST API — Calibration and Health

Base URL: `http://<server>:8000`

---

#### `POST /calibrate/start`

Start a new calibration session for a user.

**Request body**:
```json
{
  "user_id": "user_abc123",
  "label_set": ["A", "B", "C", "HELLO", "THANK_YOU"]
}
```

**Response** `200 OK`:
```json
{
  "session_id": "sess_789xyz",
  "user_id": "user_abc123",
  "labels_requested": ["A", "B", "C", "HELLO", "THANK_YOU"],
  "samples_needed_per_label": 50,
  "expires_at": "2026-02-26T03:30:00Z"
}
```

---

#### `POST /calibrate/sample`

Submit one labeled EMG feature window to the calibration session.

**Request body**:
```json
{
  "session_id": "sess_789xyz",
  "label": "A",
  "window": [0.12, -0.34, 0.05, ...]
}
```

- `window`: Float32 array of length 80 (the normalized feature vector for one window).

**Response** `200 OK`:
```json
{
  "session_id": "sess_789xyz",
  "label": "A",
  "samples_received": 12,
  "samples_needed": 50
}
```

**Error** `400 Bad Request`:
```json
{
  "error": "INVALID_WINDOW_SIZE",
  "message": "Expected feature vector of length 80, got 75."
}
```

---

#### `POST /calibrate/finish`

Finalize calibration, trigger fine-tuning, and save the user profile.

**Request body**:
```json
{
  "session_id": "sess_789xyz"
}
```

**Response** `200 OK`:
```json
{
  "session_id": "sess_789xyz",
  "user_id": "user_abc123",
  "status": "complete",
  "accuracy_per_class": {
    "A": 0.96,
    "B": 0.92,
    "C": 0.94,
    "HELLO": 0.98,
    "THANK_YOU": 0.91
  },
  "overall_accuracy": 0.942,
  "profile_saved": true,
  "profile_path": "profiles/user_abc123.json"
}
```

---

#### `GET /calibrate/profile/{user_id}`

Retrieve the calibration profile for a user.

**Response** `200 OK`:
```json
{
  "user_id": "user_abc123",
  "created_at": "2026-02-20T14:30:00Z",
  "updated_at": "2026-02-26T01:15:00Z",
  "label_set": ["A", "B", "C", "HELLO", "THANK_YOU"],
  "overall_accuracy": 0.942,
  "model_version": "1.0.0"
}
```

**Error** `404 Not Found`:
```json
{
  "error": "PROFILE_NOT_FOUND",
  "message": "No calibration profile found for user_id: user_abc123"
}
```

---

#### `DELETE /calibrate/profile/{user_id}`

Delete all calibration data for a user.

**Response** `200 OK`:
```json
{
  "user_id": "user_abc123",
  "deleted": true
}
```

---

#### `GET /health`

Server health and readiness check.

**Response** `200 OK`:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0",
  "active_ws_connections": 1,
  "uptime_seconds": 3612.4,
  "server_version": "1.0.0"
}
```

---

## 6. Mobile App Component Diagram

```
App (expo-router)
│
├── (tabs)/
│   ├── index.tsx (HomeScreen)
│   │   ├── StatusBar.tsx
│   │   │   └── Shows: BLE status, WS status, signal quality
│   │   ├── SignalMonitor.tsx
│   │   │   └── 8-channel real-time waveform display (canvas)
│   │   └── TranscriptView.tsx
│   │       └── Scrollable list of recognized labels + timestamps
│   │
│   ├── calibration.tsx (CalibrationScreen)
│   │   ├── Label prompt (e.g., "Hold sign: A")
│   │   ├── Progress bar (samples collected / samples needed)
│   │   └── Submit / Skip controls
│   │
│   └── settings.tsx (SettingsScreen)
│       ├── Server URL input (EXPO_PUBLIC_SERVER_URL override)
│       ├── BLE device selection
│       ├── TTS rate / pitch controls
│       └── Confidence threshold slider
│
├── hooks/
│   ├── useWebSocket.ts
│   │   ├── Manages ws:// connection lifecycle
│   │   ├── Handles reconnection (exponential backoff)
│   │   └── Emits {label, confidence} events
│   └── useSpeech.ts
│       ├── Wraps expo-speech
│       ├── Implements TTS queue (prevents overlapping speech)
│       └── Respects debounce from server suppressed flag
│
└── bluetooth/
    └── BLEManager.ts
        ├── Scans for BLE device with name prefix "Myo"
        ├── Writes SET_EMG_MODE command to enable 200 Hz raw streaming
        ├── Subscribes to all 4 MYO EMG notify characteristics
        └── Forwards raw int8 bytes to WebSocket send queue / on-device processor
```

---

## 7. Latency Budget Analysis

| Stage | Mechanism | Typical Latency | Notes |
|-------|-----------|----------------|-------|
| EMG sampling | MYO internal ADC (200 Hz) | < 5 ms | On-device hardware sampler |
| MYO BLE notification | BLE 5.0 connection event | ~10 ms | 4 characteristics at ~50 Hz each |
| WiFi/LAN transfer | WebSocket frame | ~2–5 ms | Local network |
| Window accumulation | Server ring buffer | 100 ms | 50% overlap step size |
| Feature extraction | NumPy vectorized | ~1 ms | 80 features, 8 ch |
| ONNX inference | CPU (no GPU needed) | ~10–20 ms | Intel/ARM server CPU |
| JSON serialization | orjson | < 0.5 ms | Fast JSON library |
| WebSocket response | LAN | ~2–5 ms | |
| TTS synthesis | expo-speech | ~40–60 ms | Platform-dependent |
| **Total** | | **~175–200 ms** | Imperceptible for conversation |

### Bottleneck Analysis

The dominant latency is the **window accumulation step** (100 ms), which is fundamental to the feature extraction approach — shorter windows reduce latency but degrade accuracy. The 200 ms window / 50% overlap choice was validated to give the best accuracy-latency tradeoff for the 26-letter ASL dataset.

If sub-100 ms total latency becomes a requirement, options include:
1. Reduce overlap to 75% (step = 50 ms) — slight accuracy reduction.
2. Use a streaming LSTM operating on 5-sample micro-windows — requires model retraining.

---

## 8. Security Considerations

### Current (Development) Posture

- WebSocket and REST servers bind to `0.0.0.0` for LAN accessibility. In production, restrict to the specific LAN interface or loopback.
- No authentication on WebSocket or REST endpoints. Any device on the LAN can connect.
- EMG data is transmitted unencrypted over the local network.

### Recommended Mitigations for Production / Clinical Use

| Risk | Mitigation |
|------|-----------|
| Unauthorized server access | Add bearer token or mTLS on WebSocket and REST |
| EMG data interception on LAN | Use `wss://` (WebSocket over TLS) with a self-signed cert |
| User profile data exposure | Encrypt `profiles/*.json` at rest using Fernet (cryptography library) |
| BLE eavesdropping | Enable BLE pairing with LE Secure Connections (LESC) in firmware |
| Model extraction | Serve model behind API; never expose ONNX file directly |
| Re-identification from EMG | Anonymize stored calibration data; use random user_id tokens |
| Server DoS | Limit max WebSocket connections (default: 4); rate-limit REST endpoints |

### Data Retention

Per the IRB protocol (`docs/data-collection-protocol.md`), raw EMG recordings are stored only for the duration of the study and deleted upon completion. Calibration profiles are stored locally on the server and are not transmitted to any cloud service.

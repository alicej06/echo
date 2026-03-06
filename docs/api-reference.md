# API Reference — EMG-ASL Inference Server

**Server version**: 1.0.0
**Base WebSocket URL**: `ws://<server-ip>:8765`
**Base REST URL**: `http://<server-ip>:8000`

All REST responses are `Content-Type: application/json`. All timestamps are Unix milliseconds (int64) unless noted.

---

## Table of Contents

- [WebSocket — `/stream`](#websocket--stream)
- [POST `/calibrate/start`](#post-calibratestart)
- [POST `/calibrate/sample`](#post-calibratesample)
- [POST `/calibrate/finish`](#post-calibratefinish)
- [GET `/calibrate/profile/{user_id}`](#get-calibrateprofileuser_id)
- [DELETE `/calibrate/profile/{user_id}`](#delete-calibrateprofileuser_id)
- [GET `/health`](#get-health)

---

## WebSocket — `/stream`

### Description

Real-time bidirectional stream. The client sends raw EMG binary frames; the server responds with JSON prediction events. One WebSocket connection = one wristband session.

### Connection URL

```
ws://<server-ip>:8765/stream
```

### Connection Procedure

1. Open a WebSocket connection to `ws://<server>:8765/stream`.
2. The server immediately sends a handshake JSON text message:
   ```json
   {
     "type": "connected",
     "server_version": "1.0.0",
     "num_classes": 36,
     "sample_rate": 200,
     "window_size_samples": 40,
     "feature_vector_size": 80,
     "confidence_threshold": 0.75
   }
   ```
3. The client begins sending binary frames at ~200 Hz.
4. The server sends JSON prediction messages each time a new sliding window completes.

### Binary Frame Format (Client → Server)

Each frame is exactly **16 bytes** containing one EMG sample across all 8 channels:

```
Offset  Length  Type    Description
------  ------  ------  -----------
0       2       int16   Channel 0, big-endian (raw 12-bit ADC value, range 0-4095)
2       2       int16   Channel 1, big-endian
4       2       int16   Channel 2, big-endian
6       2       int16   Channel 3, big-endian
8       2       int16   Channel 4, big-endian
10      2       int16   Channel 5, big-endian
12      2       int16   Channel 6, big-endian
14      2       int16   Channel 7, big-endian
```

Frames that are not exactly 16 bytes are rejected with error code 4001.

### JSON Response Schema (Server → Client)

**Prediction event** (sent every ~100 ms when the window buffer fills):

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

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"prediction"` |
| `label` | string | Top predicted ASL label (e.g., `"A"`, `"HELLO"`) |
| `label_index` | int | Index into the 36-class label list |
| `confidence` | float | Softmax probability of top class (0.0–1.0) |
| `suppressed` | bool | `true` if not spoken (below threshold or debounced) |
| `timestamp_ms` | int64 | Server Unix timestamp in milliseconds |

**Error message** (sent before connection close or as non-fatal warning):

```json
{
  "type": "error",
  "code": 4001,
  "message": "Invalid frame length: expected 16 bytes, got 12"
}
```

### WebSocket Error Codes

| Code | Meaning |
|------|---------|
| 4001 | Invalid binary frame length |
| 4002 | Model not loaded — server is not ready |
| 4003 | Too many concurrent connections (max 4) |
| 4004 | Internal inference error |
| 4005 | User calibration profile not found (when profile loading was requested) |

### curl Example (WebSocket connection is not supported by standard curl; use wscat or websocat)

```bash
# Install websocat: brew install websocat
websocat ws://192.168.1.100:8765/stream
# After connection, server sends handshake JSON.
# Binary frames must be sent programmatically (see Python client below).
```

### Python Client Example

```python
import asyncio
import struct
import websockets

SERVER_URI = "ws://192.168.1.100:8765/stream"

async def stream_emg():
    async with websockets.connect(SERVER_URI) as ws:
        # Read handshake
        handshake = await ws.recv()
        print("Handshake:", handshake)

        # Simulate sending EMG frames (replace with real BLE data)
        for _ in range(1000):
            # Pack 8 channels as int16 big-endian
            sample = [512, 490, 503, 480, 498, 511, 488, 501]
            frame = struct.pack(">8h", *sample)
            await ws.send(frame)

            # Check for prediction responses (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                print("Prediction:", msg)
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(1 / 200)  # 200 Hz

asyncio.run(stream_emg())
```

---

## POST `/calibrate/start`

Start a new calibration session for a user. Returns a session token used in subsequent calls.

### Request

```
POST /calibrate/start
Content-Type: application/json
```

```json
{
  "user_id": "user_abc123",
  "label_set": ["A", "B", "C", "HELLO", "THANK_YOU"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | Unique identifier for the user (app-generated or user-entered) |
| `label_set` | string[] | Yes | Subset of ASL labels to calibrate (1–36 labels) |

### Response `200 OK`

```json
{
  "session_id": "sess_789xyz",
  "user_id": "user_abc123",
  "labels_requested": ["A", "B", "C", "HELLO", "THANK_YOU"],
  "samples_needed_per_label": 50,
  "expires_at": "2026-02-26T03:30:00Z"
}
```

Sessions expire after 30 minutes of inactivity.

### curl Example

```bash
curl -X POST http://192.168.1.100:8000/calibrate/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_abc123", "label_set": ["A", "B", "C"]}'
```

### Python Client Example

```python
import requests

resp = requests.post(
    "http://192.168.1.100:8000/calibrate/start",
    json={"user_id": "user_abc123", "label_set": ["A", "B", "C"]},
)
session = resp.json()
session_id = session["session_id"]
print(f"Started calibration session: {session_id}")
```

---

## POST `/calibrate/sample`

Submit one labeled EMG feature window to the active calibration session.

### Request

```
POST /calibrate/sample
Content-Type: application/json
```

```json
{
  "session_id": "sess_789xyz",
  "label": "A",
  "window": [0.12, -0.34, 0.05, 0.88, -0.23, ...]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | Yes | Session token from `/calibrate/start` |
| `label` | string | Yes | ASL label for this window (must be in `label_set`) |
| `window` | float32[] | Yes | Feature vector of length 80 (normalized) |

The `window` array must contain exactly 80 float values — the normalized feature vector produced by `src/data/feature_extractor.py`. The mobile app collects 50 such windows per label by streaming through `/stream` and extracting windows on the server side, then POSTing them here.

### Response `200 OK`

```json
{
  "session_id": "sess_789xyz",
  "label": "A",
  "samples_received": 12,
  "samples_needed": 50
}
```

### Error Responses

**`400 Bad Request`** — invalid window size:
```json
{
  "error": "INVALID_WINDOW_SIZE",
  "message": "Expected feature vector of length 80, got 75."
}
```

**`400 Bad Request`** — label not in session label set:
```json
{
  "error": "INVALID_LABEL",
  "message": "Label 'Z' was not included in this calibration session's label_set."
}
```

**`404 Not Found`** — session not found or expired:
```json
{
  "error": "SESSION_NOT_FOUND",
  "message": "Calibration session 'sess_789xyz' not found or has expired."
}
```

### Python Client Example

```python
import numpy as np
import requests

# window is the output of feature_extractor.extract(emg_window)
window = np.random.randn(80).tolist()  # replace with real features

resp = requests.post(
    "http://192.168.1.100:8000/calibrate/sample",
    json={
        "session_id": session_id,
        "label": "A",
        "window": window,
    },
)
print(resp.json())
# {"session_id": "sess_789xyz", "label": "A", "samples_received": 1, "samples_needed": 50}
```

---

## POST `/calibrate/finish`

Finalize the calibration session. The server fine-tunes the model's classification head on the submitted samples and saves the resulting user profile.

### Request

```
POST /calibrate/finish
Content-Type: application/json
```

```json
{
  "session_id": "sess_789xyz"
}
```

### Response `200 OK`

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

Accuracy is computed on a 20% held-out split of the calibration samples.

### Error Response

**`400 Bad Request`** — insufficient samples:
```json
{
  "error": "INSUFFICIENT_SAMPLES",
  "message": "Label 'C' has only 23 samples; minimum required is 30 for reliable fine-tuning.",
  "samples_per_label": {"A": 50, "B": 50, "C": 23}
}
```

### curl Example

```bash
curl -X POST http://192.168.1.100:8000/calibrate/finish \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_789xyz"}'
```

### Python Client Example

```python
resp = requests.post(
    "http://192.168.1.100:8000/calibrate/finish",
    json={"session_id": session_id},
)
result = resp.json()
print(f"Overall accuracy: {result['overall_accuracy']:.1%}")
print(f"Profile saved to: {result['profile_path']}")
```

---

## GET `/calibrate/profile/{user_id}`

Retrieve the saved calibration profile metadata for a user.

### Request

```
GET /calibrate/profile/user_abc123
```

### Response `200 OK`

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

### Error Response `404 Not Found`

```json
{
  "error": "PROFILE_NOT_FOUND",
  "message": "No calibration profile found for user_id: user_abc123"
}
```

### curl Example

```bash
curl http://192.168.1.100:8000/calibrate/profile/user_abc123
```

### Python Client Example

```python
resp = requests.get("http://192.168.1.100:8000/calibrate/profile/user_abc123")
if resp.status_code == 200:
    profile = resp.json()
    print(f"Profile last updated: {profile['updated_at']}")
elif resp.status_code == 404:
    print("No profile found — user needs to calibrate.")
```

---

## DELETE `/calibrate/profile/{user_id}`

Delete all calibration data for a user, including the fine-tuned model weights stored for that user.

### Request

```
DELETE /calibrate/profile/user_abc123
```

### Response `200 OK`

```json
{
  "user_id": "user_abc123",
  "deleted": true
}
```

### Error Response `404 Not Found`

```json
{
  "error": "PROFILE_NOT_FOUND",
  "message": "No calibration profile found for user_id: user_abc123"
}
```

### curl Example

```bash
curl -X DELETE http://192.168.1.100:8000/calibrate/profile/user_abc123
```

### Python Client Example

```python
resp = requests.delete("http://192.168.1.100:8000/calibrate/profile/user_abc123")
print(resp.json())
# {"user_id": "user_abc123", "deleted": true}
```

---

## GET `/health`

Server liveness and readiness probe. Returns the server's current operational status.

### Request

```
GET /health
```

### Response `200 OK` (server ready)

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

### Response `503 Service Unavailable` (model not yet loaded)

```json
{
  "status": "starting",
  "model_loaded": false,
  "model_version": null,
  "active_ws_connections": 0,
  "uptime_seconds": 2.1,
  "server_version": "1.0.0"
}
```

### curl Example

```bash
curl http://192.168.1.100:8000/health
```

### Python Client Example

```python
import time
import requests

# Poll until server is ready
while True:
    try:
        resp = requests.get("http://192.168.1.100:8000/health", timeout=2)
        health = resp.json()
        if health["model_loaded"]:
            print(f"Server ready. Model version: {health['model_version']}")
            break
        else:
            print("Server starting, waiting...")
    except requests.ConnectionError:
        print("Server not reachable yet, retrying...")
    time.sleep(2)
```

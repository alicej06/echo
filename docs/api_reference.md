# API Reference — MAIA Inference Server

Base URL (production): `https://maia-emg-asl-production.up.railway.app`

All endpoints except `/health` require an API key when `MAIA_DISABLE_AUTH=false`.

---

## Authentication

Pass the API key in the request header:

```
X-API-Key: your-api-key
```

In development (`MAIA_DISABLE_AUTH=true`), the header is optional.

**401 response:**
```json
{"detail": "Invalid or missing API key"}
```

---

## Endpoints

### `GET /health`

Server and model status. No auth required.

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if server is running |
| `model_loaded` | boolean | `true` if ONNX model loaded successfully |

---

### `GET /info`

Model metadata and signal constants. Requires auth.

**Response `200 OK`:**
```json
{
  "model": "asl_emg_classifier",
  "version": "0.1.0",
  "classes": ["A","B","C","D","E","F","G","H","I","J","K","L","M",
               "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
  "n_classes": 26,
  "sample_rate": 200,
  "n_channels": 8,
  "window_samples": 40,
  "hop_samples": 20,
  "input_shape": [1, 40, 8],
  "output_shape": [1, 26]
}
```

---

### `POST /predict`

Single-window REST inference. Send one 200ms EMG window, receive a prediction.

**Request headers:**
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Request body:**
```json
{
  "window": [[float, float, float, float, float, float, float, float], ...]
}
```

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `window` | array of arrays | `(40, 8)` | 40 samples × 8 channels · float32 in `[-1.0, 1.0]` |

**Example request (Python):**
```python
import requests
import numpy as np

# Simulate one 200ms window of normalized sEMG
window = np.random.randn(40, 8).astype(np.float32)
window = np.clip(window * 0.1, -1.0, 1.0)

resp = requests.post(
    "https://maia-emg-asl-production.up.railway.app/predict",
    headers={"X-API-Key": "your-key"},
    json={"window": window.tolist()}
)
print(resp.json())
```

**Response `200 OK`:**
```json
{
  "letter": "A",
  "class_index": 0,
  "confidence": 0.94,
  "logits": [-2.1, 3.8, -1.5, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `letter` | string | Predicted ASL letter (`"A"` through `"Z"`) |
| `class_index` | int | Class index 0–25 |
| `confidence` | float | Softmax probability of top prediction (0–1) |
| `logits` | array[float] | Raw 26-dim output logits |

**Error responses:**
```json
// 400 — wrong input shape
{"detail": "window must be shape (40, 8), got (20, 8)"}

// 503 — model not loaded
{"detail": "Model not loaded"}
```

---

### `WS /ws/emg`

Streaming real-time inference via WebSocket. Send binary EMG frames, receive JSON predictions.

**Connection URL:**
```
wss://maia-emg-asl-production.up.railway.app/ws/emg
```

In development:
```
ws://localhost:8000/ws/emg
```

**Auth via query parameter (WebSocket doesn't support custom headers in all environments):**
```
wss://maia-emg-asl-production.up.railway.app/ws/emg?api_key=your-key
```

Or send as the first text message after connecting:
```json
{"api_key": "your-key"}
```

#### Binary frame format

Send raw sEMG windows as binary messages:

```
Frame: 640 bytes
  = 40 samples × 8 channels × 2 bytes (int16 big-endian)

Each int16 value: [-32767, +32767] → maps to [-1.0V, +1.0V]
```

**TypeScript example (React Native):**
```typescript
import { WINDOW_SAMPLES, N_CHANNELS } from '../config/constants';

function makeMockFrame(): ArrayBuffer {
  const buf = new ArrayBuffer(WINDOW_SAMPLES * N_CHANNELS * 2); // 640 bytes
  const view = new DataView(buf);
  for (let i = 0; i < WINDOW_SAMPLES * N_CHANNELS; i++) {
    const val = Math.floor((Math.random() - 0.5) * 2000);
    view.setInt16(i * 2, val, false); // big-endian
  }
  return buf;
}

ws.send(makeMockFrame());
```

**Python example:**
```python
import websockets
import numpy as np
import struct
import asyncio

async def stream():
    async with websockets.connect("ws://localhost:8000/ws/emg") as ws:
        while True:
            # 40 samples × 8 channels int16 big-endian
            window = np.random.randint(-1000, 1000, (40, 8), dtype=np.int16)
            frame = struct.pack(f">{40*8}h", *window.flatten())
            await ws.send(frame)

            response = await ws.recv()
            print(response)
            await asyncio.sleep(0.1)  # 10Hz send rate

asyncio.run(stream())
```

#### JSON response (per frame)

```json
{
  "letter": "B",
  "class_index": 1,
  "confidence": 0.87,
  "logits": [-1.2, 4.1, 0.3, ...]
}
```

#### Text ping/pong

Send the string `"ping"` to receive `"pong"` — useful for connection keep-alive.

#### Error messages (text, not binary)

```json
{"error": "Invalid frame size: expected 640 bytes, got 320"}
{"error": "Model not loaded"}
```

---

## Rate Limits

No rate limits in development. In production, Railway's free tier handles ~10 concurrent WebSocket connections comfortably. If you need higher throughput, upgrade the Railway plan or scale horizontally.

---

## Full Request Examples

### curl — health check
```bash
curl https://maia-emg-asl-production.up.railway.app/health
```

### curl — predict
```bash
curl -X POST https://maia-emg-asl-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-me" \
  -d '{"window": [[0.1, -0.2, 0.05, 0.3, -0.1, 0.2, -0.05, 0.15], ...]}'
```

### wscat — WebSocket test
```bash
npx wscat -c "wss://maia-emg-asl-production.up.railway.app/ws/emg"
```

### Python — full streaming client
```bash
python scripts/test_websocket.py \
  --url wss://maia-emg-asl-production.up.railway.app/ws/emg \
  --api-key dev-key-change-me \
  --duration 10
```

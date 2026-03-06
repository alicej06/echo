# Railway Deployment Guide

## Overview

The MAIA EMG-ASL inference server runs on Railway for:

- Always-on inference (iPhone connects without a laptop running nearby)
- Demo readiness (share a URL with anyone)
- Multi-participant data collection (server stays up during IRB sessions)

Architecture:

```
iPhone (BLE) --> EMG Armband --> On-device ONNX inference (primary)
                                --> Railway server (fallback / demo)
IYA Lab SLURM --> Train model --> Upload to Cloudflare R2 --> Railway pulls on startup
```

The Railway server is the **fallback path**. The primary path is on-device ONNX running
inside the iPhone via `useOnDeviceASL`. When the ONNX model fails to load (e.g. missing
bundle asset during development), the hook automatically opens a WebSocket to the Railway
server and streams raw EMG windows there for inference.

---

## Prerequisites

- [ ] Railway account at [railway.app](https://railway.app) -- free Hobby plan gives $5
      credit/month, which covers this workload
- [ ] Railway CLI: `npm install -g @railway/cli`
- [ ] Cloudflare account (free tier) with R2 enabled
- [ ] Docker installed locally (only needed to test the image before deploying)
- [ ] Trained ONNX model at `models/asl_emg_classifier.onnx`
      (see `docs/gpu_training_guide.md` and `scripts/slurm/train_conformer.sh`)

---

## Step 1: Set Up Cloudflare R2 (model storage)

Railway containers are ephemeral -- the filesystem resets on every deploy. The ONNX model
must be stored externally and downloaded at container startup. Cloudflare R2 is used
because it has no egress fees, a generous free tier (10 GB storage, 1 million operations/
month), and an S3-compatible API.

### 1.1 Create an R2 bucket

1. Log in to [dash.cloudflare.com](https://dash.cloudflare.com).
2. In the left sidebar, click **R2 Object Storage**.
3. Click **Create bucket**.
4. Set the bucket name to `maia-emg-asl`.
5. Choose the region closest to your users (or leave as automatic).
6. Click **Create bucket**.

### 1.2 Create an R2 API token

1. On the R2 overview page, click **Manage R2 API Tokens**.
2. Click **Create API Token**.
3. Give it a name such as `maia-emg-asl-upload`.
4. Set **Permissions** to **Object Read & Write**.
5. Under **Specify bucket(s)**, select `maia-emg-asl`.
6. Click **Create API Token**.
7. Copy and save the **Access Key ID** and **Secret Access Key** -- they are shown only once.
8. Also note the **S3-compatible endpoint**, which looks like:
   `https://<account-id>.r2.cloudflarestorage.com`

### 1.3 Enable public access for the bucket (read-only)

The Railway container needs to download the model without credentials:

1. Open the `maia-emg-asl` bucket in the R2 dashboard.
2. Click **Settings**.
3. Under **Public Access**, click **Allow Access**.
4. Note the public bucket URL, which looks like:
   `https://pub-<hash>.r2.dev`

### 1.4 Set environment variables for uploads

Add these to your local shell (e.g. `~/.zshrc` or a `.env` file never committed to git):

```bash
export R2_ACCOUNT_ID="<your cloudflare account id>"
export R2_ACCESS_KEY_ID="<your r2 access key id>"
export R2_SECRET_ACCESS_KEY="<your r2 secret access key>"
export R2_BUCKET_NAME="maia-emg-asl"
export R2_PUBLIC_URL="https://pub-<hash>.r2.dev"
```

### 1.5 Upload the trained model

After training completes, upload the ONNX file and mark it as the latest artifact:

```bash
# From the project root
python scripts/upload_artifact.py \
    --file models/asl_emg_classifier.onnx \
    --set-latest
```

The script uploads the file to R2 and writes a `latest.json` pointer object in the same
bucket. The Railway container reads `latest.json` at startup to find the current model URL.

To verify the upload:

```bash
curl https://pub-<hash>.r2.dev/latest.json
# Expected output:
# {"url": "https://pub-<hash>.r2.dev/asl_emg_classifier_<timestamp>.onnx", "sha256": "..."}
```

---

## Step 2: First Deploy to Railway

### 2.1 Log in to Railway

```bash
railway login
# Opens a browser window for OAuth. After authorizing, return to the terminal.
```

### 2.2 Link the project

From the project root (where `railway.toml` lives):

```bash
railway init
# Railway will prompt for a project name. Use: maia-emg-asl
```

If the project already exists in the Railway dashboard:

```bash
railway link
# Select the project from the list.
```

### 2.3 Set environment variables

Required variables must be set before the first deploy. Run each command, or set them in
the Railway dashboard under **Variables**:

```bash
# Model location (R2 public URL of latest.json)
railway variables set R2_LATEST_JSON_URL="https://pub-<hash>.r2.dev/latest.json"

# API authentication (comma-separated list of valid API keys)
railway variables set MAIA_API_KEYS="your-api-key-here"

# Disable auth during initial testing, then set to 0 for production
railway variables set MAIA_DISABLE_AUTH="1"

# Python settings (already in railway.toml but explicit here as a reminder)
railway variables set PYTHONUNBUFFERED="1"
railway variables set PYTHONDONTWRITEBYTECODE="1"
```

Railway automatically injects `PORT` -- the `railway.toml` already configures the start
command to use it.

### 2.4 Deploy

```bash
# From the project root
railway up
```

Railway reads `railway.toml`, builds the `Dockerfile`, pushes the image, and starts the
container. The first build takes 3-5 minutes because it compiles scipy/numpy C extensions.
Subsequent deploys use layer caching and finish in under 60 seconds.

Watch the live logs:

```bash
railway logs
```

### 2.5 Get the public URL

```bash
railway domain
# Prints: https://maia-emg-asl-production.up.railway.app
```

Or open the Railway dashboard and click **Settings > Networking > Public Domain**.

---

## Step 3: Verify the Deployment

Replace `<RAILWAY_URL>` with the URL from Step 2.5 in all commands below.

### Health check

```bash
curl https://<RAILWAY_URL>/health
# Expected:
# {"status": "ok", "model_loaded": true, "uptime_seconds": 42.1}
```

### Server info

```bash
curl https://<RAILWAY_URL>/info
# Expected (example):
# {
#   "model_version": "asl_emg_classifier_20260215_143022.onnx",
#   "n_classes": 36,
#   "labels": ["A", "B", ..., "Z", "HELLO", "THANK_YOU", ...],
#   "sample_rate": 200,
#   "window_size_ms": 200,
#   "n_channels": 8,
#   "confidence_threshold": 0.75
# }
```

### Interactive API docs

Open in a browser: `https://<RAILWAY_URL>/docs`

This is the FastAPI Swagger UI. You can test all REST endpoints here, including the
calibration and artifact endpoints.

### WebSocket streaming test

Use the included test script. Install `websockets` first if needed:

```bash
pip install websockets
```

Then run:

```bash
# 20 synthetic windows to the Railway server
python scripts/test_websocket.py \
    --url wss://<RAILWAY_URL>/stream \
    --windows 20

# Stress test: 500 windows as fast as possible
python scripts/test_websocket.py \
    --url wss://<RAILWAY_URL>/stream \
    --stress \
    --windows 500
```

Note the `wss://` prefix (WebSocket Secure) -- Railway terminates TLS so the app always
uses `wss://` for Railway URLs and `ws://` for localhost.

---

## Step 4: Connect the iPhone App

### 4.1 Set the Railway URL in the mobile app

Copy the mobile environment template:

```bash
cp mobile/react-native/.env.example mobile/react-native/.env
```

Edit `mobile/react-native/.env` and set:

```env
EXPO_PUBLIC_SERVER_URL=https://<RAILWAY_URL>
EXPO_PUBLIC_RAILWAY_URL=https://<RAILWAY_URL>
EXPO_PUBLIC_API_KEY=your-api-key-here
```

The `serverConfig` module (`src/config/serverConfig.ts`) automatically converts the
`https://` URL to `wss://` for WebSocket connections -- no manual protocol change needed.

### 4.2 Rebuild and run

```bash
cd mobile/react-native
npx expo start
```

Scan the QR code with the Expo Go app on your iPhone (or run on a simulator).

### 4.3 Test the fallback path

To force the app into server-fallback mode (bypassing on-device ONNX):

1. Open the in-app Settings screen.
2. Toggle **On-Device Only** off.
3. Verify the status chip reads **Server fallback** on the Home screen.
4. Connect to the BLE armband -- predictions should now arrive from Railway.

---

## Step 5: Update Model After Training (GPU cluster workflow)

After a training run completes on the IYA Lab SLURM cluster, the new model is
automatically exported to ONNX by `train_gpu_ddp.py`. The upload step is manual:

### 5.1 Copy model from the cluster

```bash
# From your local machine
scp <username>@<cluster-host>:/path/to/output/asl_emg_classifier.onnx models/
```

### 5.2 Upload to R2 and mark as latest

```bash
python scripts/upload_artifact.py \
    --file models/asl_emg_classifier.onnx \
    --set-latest
```

### 5.3 Trigger Railway to pull the new model

Railway will pick up the new model on the next deploy. To trigger a redeploy immediately
without any code change:

```bash
railway service restart
```

The container starts, reads `R2_LATEST_JSON_URL` to find the new model URL, downloads it,
and begins serving predictions with the updated weights. This takes about 30 seconds.

### 5.4 Verify the update

```bash
curl https://<RAILWAY_URL>/info | python -m json.tool
# Check that "model_version" reflects the new filename/timestamp.
```

---

## Environment Variables Reference

All variables are set in the Railway dashboard under **Variables**, or via `railway variables set`.

| Variable | Required | Description | Example |
|---|---|---|---|
| `PORT` | Auto | Injected by Railway. Do not set manually. | `8000` |
| `R2_LATEST_JSON_URL` | Yes | Public URL of the `latest.json` pointer in R2. | `https://pub-abc123.r2.dev/latest.json` |
| `MAIA_API_KEYS` | Recommended | Comma-separated list of valid API keys. Clients send this in `X-API-Key` header. | `key1,key2` |
| `MAIA_DISABLE_AUTH` | No | Set to `1` to skip API key validation (use during initial testing only). Default: `0`. | `0` |
| `MODEL_DIR` | No | Directory where the downloaded model is cached inside the container. Default: `models`. | `models` |
| `ONNX_MODEL_PATH` | No | Full path to the ONNX model file. Overrides `MODEL_DIR` if set. | `models/asl_emg_classifier.onnx` |
| `CONFIDENCE_THRESHOLD` | No | Predictions below this confidence are suppressed. Default: `0.75`. | `0.75` |
| `DEBOUNCE_MS` | No | Minimum milliseconds between consecutive label emissions. Default: `300`. | `300` |
| `PYTHONUNBUFFERED` | Yes | Ensures logs appear in real time in Railway. Set to `1`. | `1` |
| `PYTHONDONTWRITEBYTECODE` | Yes | Prevents `.pyc` clutter in the container. Set to `1`. | `1` |

---

## Cost Estimate

| Resource | Plan | Monthly Cost |
|---|---|---|
| Railway Hobby plan (512 MB RAM, shared CPU, 1 service) | Free credit | $5 credit/month (covers this workload) |
| Railway Pro plan (8 GB RAM, dedicated CPU) | Billed per usage | ~$20-30/month |
| Cloudflare R2 storage (up to 10 GB) | Free tier | $0 |
| Cloudflare R2 egress | No egress fees | $0 |
| Cloudflare R2 Class A operations (writes, first 1M/month) | Free tier | $0 |
| Cloudflare R2 Class B operations (reads, first 10M/month) | Free tier | $0 |

For development and IRB research sessions: the Hobby plan free credit is sufficient.
The ONNX model is approximately 5 MB. Inference runs fast on CPU because the model
operates on 80-dimensional feature vectors, not raw audio or image tensors.

For production deployments with multiple concurrent users (e.g. conference demos), upgrade
to the Pro plan for dedicated CPU and more memory headroom.

---

## Troubleshooting

### `PORT` not set / server does not start

Railway injects `$PORT` at runtime. The `railway.toml` start command already uses it:

```toml
startCommand = "python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"
```

If you override the start command manually, make sure it uses `$PORT`, not a hard-coded
port number. Railway proxies all external traffic to whatever port the container binds.

### Model not found at startup

The container logs will show:

```
[startup] R2_LATEST_JSON_URL not set -- starting without model (random weights)
```

or

```
[startup] Failed to download model from R2: 403 Forbidden
```

Check that `R2_LATEST_JSON_URL` is set correctly and the bucket has public access enabled
(Step 1.3). Re-run `python scripts/upload_artifact.py --set-latest` to regenerate the
`latest.json` pointer.

### CORS errors in browser

If you access the API from a web page (not the mobile app), add the allowed origin to the
FastAPI CORS middleware in `src/api/main.py`. The Railway domain should already be in the
allowed origins list -- check for a trailing slash mismatch.

### WebSocket 403 Forbidden

This means the API key was rejected. Verify that:

1. The `MAIA_API_KEYS` variable on Railway contains the key your client is sending.
2. The client is sending the key in the `X-API-Key` header (or as `?api_key=` query
   parameter, depending on the server implementation).
3. Alternatively, set `MAIA_DISABLE_AUTH=1` during testing.

### WebSocket connects but no predictions arrive

Check that the binary frame format matches. The server expects each message to be a raw
`Int16Array` buffer: `8 channels x 40 samples x 2 bytes = 640 bytes` per window. Verify
with `scripts/test_websocket.py --url wss://<RAILWAY_URL>/stream`. If the test script
produces predictions but the mobile app does not, the issue is in `EMGWindowBuffer` or
BLE data formatting on the phone.

### Deploy fails during Docker build (OOM or timeout)

The scipy/numpy compilation step can occasionally hit Railway's build-time memory limit.
Solutions:

1. Pin `scipy` to a prebuilt wheel version in `requirements.txt` (avoid source builds).
2. Add a `--platform linux/amd64` flag if building on Apple Silicon locally.
3. Retry the deploy -- Railway build cache usually prevents the issue on the second attempt.

### `railway logs` shows restart loops

Check for:

- Missing environment variable: look for `KeyError` or `None` being passed where a string
  is expected.
- Port conflict: should not happen on Railway (only one process runs), but confirm the
  start command uses `$PORT`.
- Out-of-memory kill: upgrade to Pro plan or reduce worker count.

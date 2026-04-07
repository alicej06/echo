# When the MYO Arrives — Day-One Checklist

Everything below is pre-built and waiting. When the **Thalmic MYO Armband**
arrives, follow this order.

---

## What You Have Right Now (No Hardware Needed)

| Component | Status | Location |
|-----------|--------|----------|
| Signal processing pipeline | Done | `src/utils/` |
| Feature extraction (80-dim) | Done | `src/utils/features.py` |
| FastAPI WebSocket server | Done | `src/api/` |
| LSTM classifier + ONNX export | Done | `src/models/` |
| Per-user calibration | Done | `src/models/calibration.py` |
| React Native app (BLE, TTS, UI) | Done | `mobile/react-native/` |
| MYO live EMG recorder | Done | `scripts/record_session.py` |
| Synthetic data generator | Done | `src/data/synthetic.py` |
| Baseline model (synthetic) | Done | `models/asl_emg_classifier.onnx` |
| NinaProDB adapter | Done | `src/data/ninapro_adapter.py` |
| E2E smoke test | Done | `scripts/smoke_test.py` |
| GRABMyo adapter | Done | `src/data/grabmyo_adapter.py` |
| Italian SL adapter | Done | `src/data/italian_sl_adapter.py` |
| Mendeley ASL adapter | Done | `src/data/mendeley_asl_adapter.py` |
| CNN-LSTM model | Done | `src/models/cnn_lstm_classifier.py` |
| SVM baseline | Done | `src/models/svm_classifier.py` |
| Docker deployment | Done | `Dockerfile + docker-compose.yml` |

---

## Datasets Available Right Now (No Hardware Needed)

These public EMG datasets can be downloaded today and used to pre-train the
model before any real user data is collected. All bridge to the session
CSV format via adapters in `src/data/`.

| Dataset | Subjects | Gestures | Channels / Rate | How to Get | Adapter |
|---------|----------|----------|-----------------|------------|---------|
| **GRABMyo** (PhysioNet) | 43 subjects, 3 sessions | 16 hand/finger + rest | 32 ch @ 2048 Hz | `wget -r -N -c -np https://physionet.org/files/grabmyo/1.1.0/` | `src/data/grabmyo_adapter.py` |
| **NinaProDB DB1** | 27 subjects | 52 hand/wrist gestures | 8 ch Myo @ 100 Hz | Register at ninapro.hevs.ch | `src/data/ninapro_adapter.py` |
| **Italian Sign Language** | 3 participants | 26 letters, 30 reps | 8 ch Myo @ 200 Hz | `git clone https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet` | `src/data/italian_sl_adapter.py` |
| **Mendeley ASL Myo** | Multiple users | ASL words | 16 ch (use 8) @ 200 Hz | data.mendeley.com/datasets/wgswcr8z24/2 | `src/data/mendeley_asl_adapter.py` |
| **ASLA (RIT)** _(pending)_ | 24 subjects | 26 letters + neutral, 40 reps | 8 ch @ 200 Hz | EULA approval pending | `src/data/asla_adapter.py` |

Quick-start:
```bash
# Download GRABMyo (~9.4 GB)
wget -r -N -c -np https://physionet.org/files/grabmyo/1.1.0/

# Convert first 10 subjects
python scripts/prepare_grabmyo.py \
    --data-dir physionet.org/files/grabmyo/1.1.0/ \
    --max-subjects 10
```

---

## Run the Full Pipeline Right Now (No Hardware)

```bash
# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Verify everything works (8/8 tests)
python scripts/smoke_test.py

# 3. Download Italian SL dataset (~50MB, instant)
bash scripts/download_datasets.sh --italian-sl
python scripts/prepare_italian_sl.py --data-dir data/external/italian-sl/

# 4. Train LSTM on Italian SL data
python scripts/train_real.py --data-dir data/raw/italian_sl/ --epochs 50

# 5. Start inference server
./start-server.sh

# 6. Test server end-to-end
python scripts/test_websocket.py

# 7. Run full pipeline demo
python scripts/demo_pipeline.py
```

---

## Step 0 — Verify Everything Works (Pre-Hardware)

```bash
cd /path/to/emg-asl-layer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Should pass 8/8 without hardware
python scripts/smoke_test.py

# Run pipeline demo with synthetic data
python scripts/demo_pipeline.py

# Start the inference server
./start-server.sh
# → REST API at  http://localhost:8000
# → WebSocket at ws://localhost:8765/stream
```

---

## Step 1 — Install MyoConnect and myo-python

**Hardware needed:** MYO Armband, Myo USB Bluetooth dongle, Mac or Windows laptop

### 1a — Install MyoConnect

Download and install **MyoConnect** from the Thalmic Labs archive:

- macOS: `MyoConnect.app` (dmg installer)
- Windows: `MyoConnect_Installer.exe`

After installing, plug in the MYO USB Bluetooth dongle and launch MyoConnect.
The menu bar icon will appear.

### 1b — Install myo-python

```bash
pip install myo-python>=0.2.1
```

Point myo-python to the Myo SDK dynamic library:

```bash
# macOS — MyoConnect bundles the SDK inside the .app
export MYO_SDK_PATH="/Applications/Myo Connect.app/Contents/Frameworks"

# Windows (default install path)
# set MYO_SDK_PATH=C:\Program Files (x86)\Myo Connect\sdk
```

Add `MYO_SDK_PATH` to your shell profile or `.env` file to persist it.

### 1c — Verify myo-python Connection

```bash
python - <<'EOF'
import myo
myo.init()
hub = myo.Hub()
print("myo-python OK — hub created")
hub.stop()
EOF
```

You should see `myo-python OK — hub created`. If MyoConnect is not running,
you will get `RuntimeError: Hub failed to start`.

---

## Step 2 — Pair and Wake the MYO

1. Launch **MyoConnect** (menu bar icon).
2. Tap the MYO logo on the armband to wake it (double-tap or hold 2 s).
3. MyoConnect will discover and pair automatically. The LED on the armband
   pulses white while pairing, then turns solid green when connected.
4. Confirm in MyoConnect's "Devices" panel that the armband appears.

---

## Step 3 — Verify EMG Stream

```bash
python - <<'EOF'
import myo, time

class Listener(myo.DeviceListener):
    def on_connected(self, event):
        print(f"[MYO] Connected: {event.device_name}")
        event.device.stream_emg(True)
    def on_emg(self, event):
        print(f"[EMG] {event.emg}")
        return False  # stop after first sample

myo.init()
hub = myo.Hub()
hub.run(1000, Listener())
EOF
```

You should see an `[EMG]` line with 8 integer values. If nothing prints,
confirm the armband is paired in MyoConnect and the SDK path is set.

---

## Step 4 — Collect Your First Real Data

Follow the IRB protocol in `docs/data-collection-protocol.md`.

Quick pilot test (just you, no IRB needed):

```bash
python scripts/record_session.py \
  --participant P001 \
  --output data/raw/ \
  --labels A B C D E F G H I J K L M N O P Q R S T U V W X Y Z \
  --reps 5
```

This records 5 reps of each of the 26 letters using myo-python and saves to
`data/raw/P001_YYYYMMDD_HHMMSS.csv`.

Electrode placement: Put the armband on the dominant forearm, 2–3 cm below the
elbow, with the MYO logo facing toward the wrist. Tighten until snug — loose
contact is the #1 cause of noisy data.

---

## Step 5 — Train on Real Data

Once you have data from 3+ participants:

```bash
# Train on real data (replaces synthetic baseline)
python scripts/train_real.py \
  --data-dir data/raw/ \
  --epochs 100 \
  --output models/

# Exports:
#   models/asl_emg_classifier.pt   (PyTorch)
#   models/asl_emg_classifier.onnx (production)
```

---

## Step 6 — Build and Run the Mobile App

The iOS app connects directly to the MYO via BLE — no MyoConnect required.

```bash
cd mobile/react-native
npm install

cp .env.example .env
# Edit .env: set EXPO_PUBLIC_SERVER_URL to your laptop's LAN IP
# The app will fall back to on-device ONNX if server is unreachable

# Development build (physical iPhone required for BLE)
npx expo run:ios --device

# Production build via EAS
eas build --platform ios --profile production
```

The app scans for BLE devices advertised as "Myo", writes the `SET_EMG_MODE`
command to enable raw 200 Hz streaming, then subscribes to all 4 EMG notify
characteristics.

---

## Step 7 — First Live Demo

1. Launch MyoConnect and confirm armband is connected (green LED).
2. Start inference server: `./start-server.sh`
3. Launch app on iPhone — tap **Connect to MYO**.
4. App will scan, find the armband, and stream raw EMG.
5. Sign the ASL alphabet — app speaks each recognized sign.
6. Run **Calibration** (30–60 s) to personalize for your forearm anatomy.

---

## Target Accuracy Milestones

| Data | Expected Accuracy |
|------|------------------|
| Synthetic baseline | ~3% (random, 36 classes) |
| 1 participant, 5 reps | ~40-60% |
| 3 participants, 10 reps | ~65-75% |
| 10 participants, 10 reps + calibration | >90% |

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Hub failed to start` | MyoConnect not running | Launch MyoConnect first |
| `myo.init()` crash | Wrong SDK path | Set `MYO_SDK_PATH` env var |
| No EMG data printed | Armband not streaming | Ensure `stream_emg(True)` is called in `on_connected` |
| Noisy / flat signal | Loose armband | Tighten until snug; re-wet electrodes with water if dry |
| BLE scan timeout on iPhone | iOS Bluetooth off | Enable Bluetooth; grant permission when prompted |
| App can't find "Myo" | Armband asleep | Double-tap MYO logo to wake; confirm green LED |

---

## Resources

- myo-python docs: https://github.com/NiklasRosenstein/myo-python
- MYO BLE protocol (reverse-engineered): https://github.com/thalmiclabs/myo-bluetooth
- myo_ecn toolbox: https://github.com/smetanadvorak/myo_ecn
- NinaProDB (public EMG dataset): https://ninapro.hevs.ch/instructions/DB1.html
- awesome-emg-data (curated dataset list): https://github.com/x-labs-xyz/awesome-emg-data

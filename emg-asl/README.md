# EMG-ASL Layer

Real-time American Sign Language recognition using surface EMG (sEMG).
Part of the MAIA Biotech project — Spring 2026.

---

## What This Is

The EMG-ASL Layer decodes ASL hand-shapes directly from forearm muscle
activity instead of a camera. The **Thalmic MYO Armband** sits just below
the elbow, reads the muscle contractions that drive each finger and thumb
through its 8 built-in electrodes, and streams raw signals at 200 Hz over
Bluetooth Low Energy. An on-device or server-side LSTM classifier turns those
signals into letter predictions in under 200 ms, and a companion iPhone app
speaks them aloud through iOS Text-to-Speech.

Hardware: Thalmic MYO Armband (8-ch sEMG, 200 Hz, int8).
Server-side connection via **myo-python** + MyoConnect (USB dongle).
Mobile app connects directly via MYO BLE protocol (no dongle required).

---

## Architecture

```
Thalmic MYO Armband            Laptop / Mac                  iPhone App (React Native)
  8-ch sEMG @ 200 Hz    -->   myo-python SDK           -->   BLE Manager (direct MYO BLE)
  forearm placement           MyoConnect + USB dongle         EMG Stream buffer
  int8 [-128, 127]            record_session.py               WebSocket client
                                                                      |
                                                           On-device ONNX inference
                                                           (or fallback to FastAPI)
                                                                      |
                                                        Bandpass filter  20-450 Hz
                                                        60 Hz notch filter
                                                        200 ms sliding window
                                                        80-dim feature vector
                                                        LSTM / CNN-LSTM / SVM
                                                                      |
                                                           {"label":"B","conf":0.91}
                                                                      |
                                                        Text-to-Speech output
                                                        Live transcript on screen
```

End-to-end latency budget:

| Stage | Time |
|---|---|
| MYO BLE notify | ~10 ms |
| Windowing (200 ms window, 50% overlap) | ~100 ms |
| ONNX inference on server CPU | ~15 ms |
| TTS phoneme synthesis | ~50 ms |
| **Total** | **~175 ms** |

---

## Getting Started (No Hardware Needed)

Everything in this section works with zero physical hardware. The full
pipeline runs on synthetic data, and real public datasets can be downloaded
in minutes.

### 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For webcam-based pose collection (Vision-to-EMG auto-labeling), also install:

```bash
pip install 'mediapipe>=0.10' 'opencv-python>=4.9'
```

### 2. Run Smoke Tests

```bash
python scripts/smoke_test.py   # should print 8/8 PASS
```

### 3. Download a Dataset

```bash
bash scripts/download_datasets.sh --italian-sl   # ~50 MB, instant
python scripts/prepare_italian_sl.py --data-dir data/external/italian-sl/
```

### 4. Train and Run

```bash
# Train LSTM on Italian SL data (26 letters, 8-ch Myo @ 200 Hz)
python scripts/train_real.py --data-dir data/raw/italian_sl/ --epochs 50

# Start inference server
./start-server.sh
# WebSocket at  ws://localhost:8765/stream
# REST API at   http://localhost:8000

# Verify server end-to-end
python scripts/test_websocket.py

# Run full pipeline demo with synthetic data
python scripts/demo_pipeline.py
```

---

## Datasets

All datasets below bridge to the shared session CSV format through adapter
scripts in `src/data/`. Adapters normalize channel count, sample rate, and
label format so `train_real.py` consumes any of them identically.

| Dataset | Size | Channels / Rate | Subjects | Gestures | Access | Adapter |
|---|---|---|---|---|---|---|
| **GRABMyo** (PhysioNet) | ~9.4 GB | 32 ch @ 2048 Hz | 43 subjects, 3 sessions | 16 hand/finger + rest | Free direct download | `src/data/grabmyo_adapter.py` |
| **NinaProDB DB1** | ~2 GB | 8 ch Myo @ 100 Hz | 27 subjects | 52 hand/wrist gestures | Free, academic registration | `src/data/ninapro_adapter.py` |
| **Italian Sign Language** | ~50 MB | 8 ch Myo @ 200 Hz | 3 participants | 26 letters, 30 reps | `git clone` from GitHub | `src/data/italian_sl_adapter.py` |
| **Mendeley ASL Myo** | ~300 MB | 16 ch @ 200 Hz (use 8) | Multiple users | ASL words | Free browser download | `src/data/mendeley_asl_adapter.py` |
| **ASLA (RIT)** _(pending)_ | ~1.5 GB | 8 ch @ 200 Hz | 24 subjects | 26 letters + neutral, 40 reps | EULA approval pending | `src/data/asla_adapter.py` |

Recommended starting point: **Italian Sign Language** — 50 MB, instant
`git clone`, same 8-ch / 200 Hz Myo setup, 26 ASL letters.

For pre-training breadth: **GRABMyo** — 43 subjects, free direct download,
covers open-hand, fist, and pinch shapes that map naturally to ASL letters.

See [data/README.md](data/README.md) for the full dataset catalogue including
additional community datasets from the `awesome-emg-data` list.

---

## Training on the IYA GPU Cluster

The MAIA team trains on the **IYA Nvidia Lab cluster** (18x RTX A6000, 48 GB
each). The same cluster is used for the Nalana project. See
[docs/gpu_training_guide.md](docs/gpu_training_guide.md) for the full setup
guide (SSH, conda env, data transfer, multi-GPU torchrun commands, W&B
logging).

Quick start on the cluster:

```bash
# Single GPU, full Italian SL, ~3 min
CUDA_VISIBLE_DEVICES=0 python scripts/train_real.py \
    --data-dir data/raw/italian_sl/ --epochs 50 --batch-size 2048 --amp

# 4-GPU DDP, all datasets, ~10 min
torchrun --nproc_per_node=4 scripts/train_gpu_ddp.py \
    --model lstm --data-dir data/raw/ --epochs 200 --batch-size 8192 --amp \
    --wandb --wandb-project maia-emg-asl
```

---

## Models

Three classifiers are implemented and ready to train. All three export to
ONNX for production inference and accept the same 80-dimensional feature
vector (5 time-domain + 5 frequency-domain features per channel across 8
channels).

| Model | Architecture | Expected Accuracy | Best For |
|---|---|---|---|
| **LSTM** | 2-layer LSTM, hidden=128, dropout=0.3 | 65-85% (real data, 10+ subjects) | Primary production model |
| **CNN-LSTM** | 1D Conv feature extractor + LSTM | 70-88% (real data, 10+ subjects) | Higher data regimes |
| **SVM baseline** | RBF kernel, feature-engineered input | 55-75% (real data, 5+ subjects) | Quick baseline, no GPU needed |

Training scripts:

```bash
python scripts/train_real.py --data-dir data/raw/ --epochs 100        # LSTM
python scripts/train_cnn_lstm_baseline.py --data-dir data/raw/         # CNN-LSTM
python scripts/train_svm_baseline.py --data-dir data/raw/              # SVM
```

Accuracy milestones (all models):

| Data available | Expected accuracy |
|---|---|
| Synthetic baseline | ~3% (random, 36 classes) |
| 1 participant, 5 reps | 40-60% |
| 3 participants, 10 reps | 65-75% |
| 10 participants, 10 reps + calibration | >90% |

---

## Vision-to-EMG Auto-Labeling

`src/data/vision_teacher.py` implements a cross-modal labeling pipeline that
eliminates manual annotation. Record a short webcam video alongside a MYO EMG
session, then run MediaPipe Hands on the video to extract per-frame ASL letter
predictions. Those predictions are synced to EMG timestamps using
nearest-neighbor matching (50 ms tolerance), producing a fully labeled
training CSV at zero annotation cost.

### Train the Pose Classifier

```bash
# Record 5 seconds of each letter A-Z from your webcam:
python scripts/train_pose_classifier.py --collect

# Fit the classifier from saved landmarks:
python scripts/train_pose_classifier.py --train

# Or do both in one step:
python scripts/train_pose_classifier.py --collect --train

# Evaluate an existing classifier (confusion matrix + F1):
python scripts/train_pose_classifier.py --evaluate
```

The trained classifier is saved to `models/pose_classifier.joblib`.

### Run the Auto-Labeler

```bash
python scripts/auto_label_session.py \
  --webcam \
  --classifier models/pose_classifier.joblib \
  --emg-session data/raw/P001_20260227.csv \
  --output data/processed/P001_20260227_labeled.csv
```

---

## Hardware

See [HARDWARE_ARRIVES.md](HARDWARE_ARRIVES.md) for the MYO day-one checklist:
MyoConnect setup, myo-python installation, stream verification, and data
collection commands.

See [docs/hardware-setup.md](docs/hardware-setup.md) for the full MYO armband
setup guide including electrode placement and SDK installation.

See [hardware/myo_armband/README.md](hardware/myo_armband/README.md) for
MYO-specific BLE protocol documentation and troubleshooting.

---

## Mobile App

The React Native app in `mobile/react-native/` handles the full user-facing
experience:

- **BLE** — connects directly to the MYO Armband via the reverse-engineered
  MYO BLE GATT protocol (no MyoConnect required on mobile); subscribes to all
  4 EMG notify characteristics to reconstruct 8-channel @ 200 Hz
- **On-device inference** — an ONNX Runtime session runs the classifier
  locally with no server required; the FastAPI WebSocket server is the fallback
  for CPU inference on a laptop
- **Text-to-Speech** — `expo-speech` pronounces each recognized letter or
  word immediately after the debounce window clears
- **Calibration** — a 30-60 second guided session fine-tunes a linear
  adaptation layer for the current user's forearm anatomy (+10-15 pp accuracy)

```bash
cd mobile/react-native
npm install
cp .env.example .env
# Set EXPO_PUBLIC_SERVER_URL to your laptop's LAN IP in .env
npx expo run:ios --device   # physical iPhone required for BLE
```

---

## Project Structure

```
emg-asl-layer/
├── README.md                         This file
├── HARDWARE_ARRIVES.md               MYO day-one checklist
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── start-server.sh
│
├── src/
│   ├── api/                          FastAPI WebSocket + REST server
│   ├── data/
│   │   ├── vision_teacher.py         Cross-modal auto-labeling pipeline
│   │   ├── synthetic.py              Synthetic EMG data generator
│   │   ├── recorder.py               MYO live EMG recorder (myo-python)
│   │   ├── grabmyo_adapter.py
│   │   ├── ninapro_adapter.py
│   │   ├── italian_sl_adapter.py
│   │   ├── mendeley_asl_adapter.py
│   │   └── asla_adapter.py
│   ├── models/
│   │   ├── lstm_classifier.py
│   │   ├── cnn_lstm_classifier.py
│   │   ├── svm_classifier.py
│   │   └── calibration.py
│   └── utils/
│       ├── constants.py              Shared constants (MYO UUIDs, signal params)
│       ├── features.py               80-dim feature extraction
│       ├── filters.py                Butterworth + notch filters
│       └── pipeline.py              End-to-end processing pipeline
│
├── scripts/
│   ├── smoke_test.py
│   ├── train_pose_classifier.py
│   ├── auto_label_session.py
│   ├── train_real.py
│   ├── train_cnn_lstm_baseline.py
│   ├── train_svm_baseline.py
│   ├── record_session.py             Live MYO EMG recorder (myo-python)
│   ├── demo_pipeline.py
│   ├── download_datasets.sh
│   └── prepare_*.py
│
├── mobile/react-native/              React Native iOS/Android app
├── hardware/myo_armband/             MYO BLE protocol docs & setup guide
├── models/                           Trained model files (gitignored)
├── data/
│   ├── pose_landmarks/
│   ├── raw/                          EMG session CSVs (gitignored)
│   └── processed/                    Labeled + windowed features (gitignored)
├── notebooks/
├── tests/
└── docs/                             Architecture, hardware setup, API reference
```

---

## License and Citation

MIT License. See `LICENSE` for the full text.

If you use this system or any part of this codebase in research, please cite:

```
Newton, C. (2026). MAIA Biotech EMG-ASL Layer. MAIA Biotech.
```

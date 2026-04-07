# Changelog

All notable changes to MAIA EMG-ASL are documented here.

---

## [0.1.0] — 2026-02-27

### Added

**Core ML pipeline**
- LSTM classifier (246K params) — on-device primary model
- CNN-LSTM classifier — hybrid spatial + temporal architecture
- Conformer classifier (~1.5M params) — server / best-accuracy model
- SVM baseline (scikit-learn) — classic ML reference
- CLIP-style cross-modal embedding (EMGEncoder + VisionEncoder, symmetric InfoNCE)
- Signal processing: bandpass 20–450 Hz, notch 60 Hz, full-wave rectification, sliding window
- 80-dim feature extraction: RMS, MAV, WL, ZC, SSC, VAR, AR(4), IEMG, kurtosis, MNF × 8 channels
- EMG data augmentation: time-warp, channel dropout, amplitude scaling, Gaussian noise
- Vision teacher: MediaPipe Hands → 63-dim wrist-relative L2-normalized landmarks

**Inference server**
- FastAPI inference server with `/health`, `/info`, `/predict`, `/ws/emg` endpoints
- WebSocket streaming: 640-byte binary int16 big-endian frames → JSON predictions
- API key authentication middleware (X-API-Key header)
- Deployed to Railway with native GitHub auto-deploy integration

**Training infrastructure**
- `train_lstm_baseline.py` — synthetic data training + ONNX export (CPU-friendly, ~30s)
- `train_conformer_baseline.py` — Conformer on synthetic data
- `train_gpu_ddp.py` — PyTorch DDP multi-GPU training via `torchrun`
- `train_cross_modal.py` — EMG↔vision cross-modal embedding
- `optuna_hpo.py` — Optuna hyperparameter search
- SLURM job scripts for IYA Nvidia Lab: LSTM, Conformer, cross-modal, HPO
- Cloudflare R2 artifact upload/download scripts

**Data pipeline**
- `calibrate_user.py` — per-user normalization baseline
- `auto_label_session.py` — vision teacher auto-labeling pipeline
- `validate_session.py` — session quality checker with `--fix` mode
- `download_wlasl.py` — WLASL dataset download and extraction
- `export_coreml.py` — ONNX → CoreML `.mlpackage` for iOS Neural Engine
- `benchmark_models.py` — latency + accuracy comparison

**Mobile app** (React Native / Expo 53)
- expo-router v5 tab layout: ASL Live, Demo, Settings
- ASL Live screen: WebSocket connection, mock EMG frame generator (big-endian int16), animated letter display, confidence bar, history
- Demo screen: alphabet cycling at 0.8s/letter, server health check
- Settings screen: server URL, API key, BLE device name, debug mode
- Dark theme throughout
- Pre-configured `.env` pointing to live Railway server

**Documentation**
- `README.md` — full project overview with architecture, quickstart, iOS setup, Railway guide
- `docs/api_reference.md` — REST + WebSocket API reference with full examples
- `docs/railway_deployment.md` — Railway + R2 deployment guide
- `docs/gpu_training_guide.md` — IYA Nvidia Lab + SLURM training guide
- `docs/model_card.md` — model specs, benchmarks, limitations, citations
- `docs/data-collection-protocol.md` — sEMG recording protocol, electrode placement, IRB guidance
- `docs/hardware-setup.md` — MAIA Band setup, BLE spec, calibration, troubleshooting
- `CONTRIBUTING.md` — branch conventions, code style, PR process
- `CHANGELOG.md` — this file

**Verification**
- 21 pytest tests covering: signal filters, feature extraction, model forward pass, ONNX export, API endpoints, WebSocket protocol
- Baseline ONNX model (8.2KB) trained, verified, and committed to repo
- Railway live: `https://maia-emg-asl-production.up.railway.app/health`

### Known limitations in 0.1.0

- Training data is synthetic only — real sEMG accuracy unknown until hardware arrives
- Static ASL letters A–Z only; no dynamic signs or words
- Single-user, right-hand only; not validated cross-user
- CoreML export script exists but has not been run on a real device yet

---

## Roadmap

### [0.2.0] — Hardware phase (ETA: when MAIA Band arrives)
- [ ] Real sEMG data collection (5+ sessions, 1 user)
- [ ] Per-user calibration pipeline validated on real data
- [ ] Retrain LSTM + Conformer on real data — target > 95% single-session accuracy
- [ ] CoreML integration tested on physical iPhone
- [ ] BLE connection in React Native app (replace mock frame generator)

### [0.3.0] — Multi-user
- [ ] 3–5 participants data collection
- [ ] Cross-modal pre-training from WLASL → improved generalization
- [ ] Cross-user accuracy > 80%
- [ ] App Store TestFlight build

### [1.0.0] — Production
- [ ] > 95% cross-session accuracy (single user)
- [ ] > 85% cross-user accuracy
- [ ] Dynamic sign support (words and phrases)
- [ ] On-device CoreML model auto-update from R2

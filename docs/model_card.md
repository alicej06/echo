# Model Card: EMG-ASL Gesture Classifier

## Model Details

| Field | Value |
|-------|-------|
| Model name | EMG-ASL LSTM Classifier |
| Version | 1.0.0-synthetic (baseline) |
| Architecture | LSTM with input projection |
| Input | 320-dim float32 (40 samples x 8 EMG channels, flattened) |
| Output | 36-class logits (A-Z + 10 common words) |
| Parameters | ~246K |
| Framework | PyTorch 2.x, ONNX Runtime 1.17+ |
| Created | February 2026 |
| Organization | MAIA Biotech |

---

## Intended Use

**Primary use:** Real-time ASL gesture recognition from surface EMG (sEMG) signals
recorded from the forearm using the **Thalmic MYO Armband** (8-channel sEMG,
200 Hz, int8). Server-side connection via myo-python + MyoConnect; mobile app
connects directly via MYO BLE GATT protocol.

**Target users:** Researchers studying assistive communication technologies and
developers building sign language recognition systems.

**Not intended for:** Medical diagnosis, safety-critical applications, or users
with neuromuscular conditions requiring clinical-grade accuracy.

---

## Training Data

> **IMPORTANT: The current baseline model (v1.0.0-synthetic) is trained exclusively
> on procedurally generated synthetic data. It does not generalize to real EMG
> signals and should be treated as a structural placeholder only.**

| Dataset | Subjects | Gestures | Approx. Hours | Notes |
|---------|----------|----------|---------------|-------|
| Synthetic (baseline) | N/A | 36 classes (A-Z + 10 words) | < 1 min generate time | Gaussian noise per class; NOT real EMG |
| GRABMyo (PhysioNet) | 43 subjects, 3 sessions | 16 hand/finger gestures + rest | ~40 hrs | 32 ch sEMG @ 2048 Hz; adapter in `src/data/grabmyo_adapter.py` |
| NinaProDB DB1 | 27 subjects | 52 hand/wrist gestures | ~14 hrs | 8 ch Myo @ 100 Hz; requires free academic registration |
| Italian Sign Language (GitHub) | 3 simulated participants | 26 letters, 30 reps | ~3 hrs | 8 ch Myo @ 200 Hz; direct download |
| Mendeley ASL Myo | Multiple users | ASL words | TBD | 16 ch (downsampled to 8); free browser download |
| ASLA (RIT) | 24 subjects | 26 letters + neutral, 40 reps | ~20 hrs | 8 ch @ 200 Hz; pending EULA approval |

All public dataset adapters normalize signals to the 8-channel, 200 Hz session CSV
format expected by the training pipeline (`scripts/train_real.py`).

---

## Evaluation

Accuracy figures below are empirical milestones derived from the project roadmap.
Results for synthetic and early real-data stages reflect expected ranges, not
guarantees; EMG signals vary substantially across individuals.

| Dataset / Stage | Accuracy | Notes |
|-----------------|----------|-------|
| Synthetic baseline (v1.0.0) | ~3% | Random weights, chance level for 36 classes |
| 1 participant, 5 reps per sign | ~40-60% | Expected range after first real recording session |
| 3 participants, 10 reps per sign | ~65-75% | After cross-subject LSTM training |
| 10 participants, 10 reps + per-user calibration | >90% | With per-user adaptor layer enabled |
| WLASL cross-modal (GPU server) | TBD | Zero-shot on full WLASL vocabulary; planned milestone |

Evaluation protocol: stratified 80/20 train-test split per participant,
5-fold cross-validation when N participants >= 5.

---

## Limitations

**Forearm anatomy variability.** Muscle placement and forearm diameter differ
significantly across individuals. A model trained on one person's data may
perform poorly on another without recalibration.

**Electrode placement sensitivity.** A shift of even 1-2 cm in electrode position
changes the recorded signal enough to cause misclassification. The collection
protocol (`docs/data-collection-protocol.md`) specifies a reference landmark
(2-3 cm below the elbow, circumferential spacing), but reproducibility requires
practice.

**Fatigue effects.** EMG amplitude and spectral content shift as the forearm
muscles fatigue. Sessions longer than ~20 minutes may see accuracy degrade
without a fatigue-compensation layer (not yet implemented).

**Static letters only.** The current 36-class vocabulary covers fingerspelled
ASL letters (static hand shapes) and 10 common words. Dynamic signs that require
wrist or arm motion are outside the current scope.

**Right-dominant arm assumption.** All training data uses the dominant (right)
forearm. Left-handed users must collect separate training data; the model does
not currently mirror or adapt automatically.

**MYO BLE throughput.** The MYO Armband streams 8 channels at 200 Hz across 4
BLE characteristics (~3.2 kB/s). BLE packet scheduling can introduce jitter;
the pipeline handles this but performance degrades above ~2% packet loss.

**No dynamic vocabulary expansion at runtime.** Adding a new sign class requires
retraining the final classification head. The per-user calibration adaptor
(`src/models/calibration.py`) adjusts class boundaries but does not add new
output nodes.

---

## Privacy and Ethics

**Data collection consent.** Any recording session involving human participants
requires IRB approval (protocol template: `docs/data-collection-protocol.md`).
Pilot self-recordings (single developer, no third parties) are exempt at most
institutions, but confirm with your IRB office.

**No personally identifiable information (PII) collected.** Session files record
only 8-channel EMG time series, a participant ID code (e.g., P001), and a label
sequence. No names, ages, or demographic data are stored in the session CSV.

**Withdrawal.** Participants may withdraw data at any time; their session files
can be deleted from `data/raw/` without retraining the model from scratch
(use `scripts/train_real.py --exclude PXXX`).

**Local storage only.** All session data is stored on the researcher's local
machine. Nothing is uploaded to a cloud service unless the researcher explicitly
configures a remote storage backend. The inference server does not log raw EMG
streams to disk during live use.

**Bias considerations.** The current public datasets used for pre-training
(GRABMyo, NinaProDB) over-represent male participants with typical forearm
musculature. Performance on participants outside that distribution is unknown
until additional diverse data is collected.

---

## Technical Specifications

### Signal Processing Pipeline

| Stage | Parameter | Value |
|-------|-----------|-------|
| Sample rate | `SAMPLE_RATE` | 200 Hz |
| Channels | `N_CHANNELS` | 8 (MYO Armband sEMG electrodes) |
| Window length | `WINDOW_SIZE_MS` | 200 ms (40 samples) |
| Window step | `STEP_SIZE_SAMPLES` | 20 samples (50% overlap) |
| Bandpass filter | `BANDPASS_LOW / HIGH` | 20-450 Hz (4th-order Butterworth) |
| Notch filter | `NOTCH_FREQ` | 60 Hz (IIR, power-line rejection) |

### Feature Extraction (80-dim vector per window)

Per channel (8 channels x 10 features = 80 total):

- **Time-domain (5 per channel):** Root Mean Square (RMS), Mean Absolute Value (MAV),
  Waveform Length (WL), Zero Crossings (ZC), Slope Sign Changes (SSC)
- **Frequency-domain (5 per channel):** Mean frequency, median frequency,
  2nd spectral moment, 3rd spectral moment, 4th spectral moment

Feature extractor: `src/utils/features.py`

### Model Architecture

```
Input: (batch, 320)  -- flattened 40 x 8 window
  |
  Linear(320, 128)  -- input projection
  ReLU
  Reshape to (batch, 40, 128/8)  -- back to sequence form for LSTM
  |
  LSTM(hidden=128, layers=2, dropout=0.3)
  |
  Linear(128, 36)  -- classification head
  |
Output: (batch, 36)  -- raw logits; softmax applied at inference
```

During inference, a prediction is suppressed if `max(softmax(logits)) < 0.75`
(see `CONFIDENCE_THRESHOLD` in `src/utils/constants.py`). A 300 ms debounce
window (`DEBOUNCE_MS`) prevents duplicate label emissions for held signs.

### ONNX Export

The model is exported with `opset=17` and tested against ONNX Runtime 1.17+.
The ONNX graph has a single float32 input node named `input` with shape
`[batch_size, 320]` and a single float32 output node named `output` with shape
`[batch_size, 36]`.

Dynamic axes are enabled for `batch_size` to support both single-window
real-time inference and batched evaluation.

### Inference Server

- REST endpoint: `POST http://localhost:8000/predict`
- WebSocket stream: `ws://localhost:8765/stream`
- Expected round-trip latency: < 20 ms on M-series MacBook; < 50 ms over LAN
- Docker image: see `Dockerfile` and `docker-compose.yml`

---

## Citation

If you use this model or pipeline in published work, please cite:

```
MAIA Biotech (2026). EMG-ASL LSTM Classifier (v1.0.0-synthetic).
  GitHub: [repository URL]. February 2026.
```

For the public datasets used in pre-training, additionally cite:

- **GRABMyo:** Pradhan A. et al. (2022). GRABMyo. PhysioNet.
  https://doi.org/10.13026/cdrp-a787
- **NinaProDB DB1:** Atzori M. et al. (2014). Electromyography data for
  non-invasive naturally-controlled robotic hand prostheses. Scientific Data.
  https://doi.org/10.1038/sdata.2014.53
- **Italian Sign Language EMG:** Airtlab (2021).
  https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet

---

## Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0-synthetic | Feb 2026 | Initial baseline; trained on synthetic data only |
| 1.1.0 | TBD | First real-data training (Italian SL + GRABMyo pre-train) |
| 2.0.0 | TBD | Multi-participant LSTM + per-user calibration adaptor |

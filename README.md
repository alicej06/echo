# Echo

ASL-to-English live translation via the Myo armband + Claude.

```
Myo BLE  →  LSTM classifier  →  letters  →  Claude Haiku  →  natural English
```

Wear the armband. Fingerspell. Pause. Get natural English back.

---

## Quickstart

```bash
git clone https://github.com/alicej06/echo.git
cd echo
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/live_translate.py --scan          # find your Myo's BLE name
python scripts/live_translate.py --device "My Myo"
```

---

## How It Works

The **Thalmic Myo armband** sits just below your elbow and reads 8 forearm
muscle channels at 200 Hz over Bluetooth Low Energy. A sliding 200ms window
of filtered EMG feeds an LSTM that outputs a letter prediction (A-Z) with a
confidence score. When you pause signing for ~2 seconds, the accumulated
letters are sent to an LLM which reconstructs them into natural English.

No camera. No dongle. No MyoConnect. Pure BLE on modern macOS.

**LLM backends (automatic priority order):**

- **Claude Haiku** — best quality, set `ANTHROPIC_API_KEY`
- **Ollama cloud** — free hosted models, set `OLLAMA_API_KEY`
- **Ollama local** — runs `llama3.2` on your machine, no key needed
- **`--no-llm`** — letters only, no sentence reconstruction

---

## Personal Calibration (Recommended)

The base model was trained on multiple people. Fine-tuning it on your arm
takes about 5 minutes and significantly improves accuracy.

```bash
python scripts/calibrate_quick.py --user me
python scripts/live_translate.py --device "My Myo" --model models/calibrated/me/model.pt
```

| Model             | Accuracy (typical) |
| ----------------- | ------------------ |
| Base model        | ~55-65%            |
| After calibration | ~80-90%            |

---

## Command Reference

### `live_translate.py`

| Flag            | Default                        | Description                              |
| --------------- | ------------------------------ | ---------------------------------------- |
| `--device NAME` | `"Myo"`                        | BLE name of your Myo armband             |
| `--scan`        | —                              | List nearby BLE devices and exit         |
| `--model PATH`  | `models/asl_emg_classifier.pt` | Path to `.pt` model weights              |
| `--no-llm`      | —                              | Skip LLM entirely, show raw letters only |

### `calibrate_quick.py`

| Flag               | Default | Description                  |
| ------------------ | ------- | ---------------------------- |
| `--user ID`        | `me`    | User ID for saving the model |
| `--letters ABC...` | `A-Z`   | Letters to calibrate         |
| `--reps N`         | `3`     | Repetitions per letter       |

---

## Repo Structure

```
echo/
├── scripts/         # live_translate.py, calibrate_quick.py, training scripts
├── src/             # LSTM model, signal processing, constants
├── models/          # Model weights (ONNX base + calibrated .pt files)
├── mobile/          # React Native app (Expo, BLE + on-device ONNX inference)
├── frontend/        # Next.js web UI (live letter display, sentence history)
├── configs/         # YAML training configs
├── hardware/        # Myo armband docs and BLE protocol reference
├── notebooks/       # Jupyter notebooks for data exploration and training
├── tests/           # pytest suite
├── docs/            # API reference, architecture, deployment guides
├── requirements.txt      # Runtime deps (live inference)
└── requirements-dev.txt  # Training pipeline + dev tools
```

---

## What Echo Is

Echo is infrastructure for communities to own the language they invent.

Every generation creates language faster than any institution can document it.
Every friend group has expressions that exist nowhere in writing and vanish when
the group drifts apart. Echo makes that ownership legible, permanent, and
transferable.

The ASL community is the anchor use case: signing communities evolve language
faster than any centralized body can track. Echo gives them the infrastructure
for rapid community-level language creation and adoption.

---

All glory to God! ✝️❤️

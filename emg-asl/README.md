# Echo — EMG-ASL Live Translation

Real-time ASL fingerspelling recognition via the Myo armband + Claude.

```
Myo BLE  →  LSTM classifier  →  letters  →  Claude Haiku  →  natural English
```

---

## How It Works

The **Thalmic Myo armband** sits just below your elbow and reads 8 forearm
muscle channels at 200 Hz over Bluetooth Low Energy. A sliding 200ms window
of filtered EMG feeds an LSTM that outputs a letter prediction (A-Z) with a
confidence score. When you pause signing for ~2 seconds, the accumulated
letters are sent to Claude Haiku which reconstructs them into natural English.

No camera. No dongle. No MyoConnect. Pure BLE on modern macOS.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/alicej06/echo.git
cd echo/emg-asl
pip install -r requirements.txt
```

> Python 3.10+ required. Install `bleak`, `torch`, `scipy`, `numpy`, `anthropic`.

### 2. Set your API key (for sentence reconstruction)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Skip this step if you only want raw letter output (use `--no-llm`).

### 3. Pair your Myo

Open **System Settings > Bluetooth** and pair your Myo armband. It should
appear as a named device (e.g. "Myo", "My Myo", or your custom name).

### 4. Find your Myo's BLE name

```bash
python scripts/live_translate.py --scan
```

This scans for 8 seconds and prints all nearby BLE devices. Your Myo will be
marked with a green arrow. Note the exact name.

```
  Address                                   Name
  ----------------------------------------  ------------------------------
  E4:3B:71:9A:02:1F                         My Myo  <-- likely your Myo
```

### 5. Run live translation

```bash
python scripts/live_translate.py --device "My Myo"
```

Replace `"My Myo"` with whatever name appeared in the scan above.

Put the armband on, start fingerspelling — letters appear as you sign, and
natural English sentences appear after each pause.

---

## Personal Calibration (Recommended)

The base model was trained on multiple people. Fine-tuning it on your arm
takes about 5 minutes and significantly improves accuracy.

```bash
python scripts/calibrate_quick.py --user me
```

This walks you through signing each letter 3 times (3 seconds each), then
fine-tunes the model's output layer on your data. Saves a personal model to
`models/calibrated/me/model.pt`.

Run live translation with your personal model:

```bash
python scripts/live_translate.py --device "My Myo" --model models/calibrated/me/model.pt
```

Calibration options:

```bash
# Only calibrate a subset of letters (faster):
python scripts/calibrate_quick.py --user me --letters ABCDEFGHIJ

# More reps per letter (more accurate):
python scripts/calibrate_quick.py --user me --reps 5
```

Typical accuracy improvement:

| Model              | Accuracy (typical) |
| ------------------ | ------------------ |
| Base model         | ~55-65%            |
| After calibration  | ~80-90%            |

---

## Command Reference

### `live_translate.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--device NAME` | `"Myo"` | BLE name of your Myo armband |
| `--scan` | — | List nearby BLE devices and exit |
| `--model PATH` | `models/asl_emg_classifier.pt` | Path to model weights |
| `--no-llm` | — | Skip Claude, show raw letters only |

### `calibrate_quick.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--user ID` | `me` | User ID for saving the model |
| `--letters ABC...` | `A-Z` | Letters to calibrate |
| `--reps N` | `3` | Repetitions per letter |

---

## Architecture

```
emg-asl/
├── scripts/
│   ├── live_translate.py       # Real-time BLE -> LSTM -> LLM pipeline
│   └── calibrate_quick.py      # Personal calibration (5 min)
├── src/
│   ├── constants.py            # Sample rate, window size, BLE UUIDs
│   └── models/
│       └── lstm_classifier.py  # 2-layer LSTM, input (B, 40, 8), output (B, 26)
├── models/
│   ├── asl_emg_classifier.pt   # Base model weights
│   └── calibrated/             # Personal models saved here
└── requirements.txt
```

**Signal pipeline:**
1. BLE GATT notifications deliver 16 bytes (2 samples x 8 channels) at ~200Hz
2. Samples buffer into a rolling 4-second deque
3. Every 100ms (20 samples), a 200ms window (40 samples) is extracted
4. Butterworth bandpass filter 20-95Hz removes motion artifact and noise
5. LSTM processes the window and outputs softmax probabilities over 26 classes
6. Letter emitted if confidence >= 0.75 and debounce window (300ms) has passed
7. After 1.8s pause in signing, pending letters sent to Claude Haiku
8. Claude reconstructs letters into natural English, handling recognition errors

---

## Troubleshooting

**"not found" — device not visible**
Run `--scan` to find your Myo's actual BLE name, then pass it with `--device`.
Make sure Bluetooth is on and the armband is awake (LED blinking).

**"enable cmd failed" — shows in yellow but continues**
Normal on some firmware versions. The script continues; EMG should still stream.

**Letters wrong or random**
First run with the base model will have moderate accuracy. Run calibration:
```bash
python scripts/calibrate_quick.py --user me
```

**"No module named 'bleak'"**
```bash
pip install bleak anthropic scipy numpy torch
```

**macOS Bluetooth permission denied**
On first run macOS will prompt for Bluetooth access. Allow it. If you missed
the prompt: System Settings > Privacy & Security > Bluetooth > add Terminal (or
your Python environment).

**Myo keeps disconnecting**
Make sure the armband is seated firmly on your forearm (not the wrist). The
LED should glow white/teal when connected.

---

All glory to God! ✝️❤️

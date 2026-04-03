# Echo

ASL-to-English live translation via the Myo armband + Claude.

```
Myo BLE  →  LSTM classifier  →  letters  →  Claude Haiku  →  natural English
```

Wear the armband. Fingerspell. Pause. Get natural English back.

---

## Live Translation Pipeline (Start Here)

Everything you need to go from clone to working demo is in the `emg-asl/` folder:

**[emg-asl/README.md](emg-asl/README.md)**

Covers: install, pairing your Myo, finding its BLE name, running the live
pipeline, personal calibration, command reference, and troubleshooting.

**Quick version:**

```bash
git clone https://github.com/alicej06/echo.git
cd echo/emg-asl
pip install -r requirements-live.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/live_translate.py --scan          # find your Myo's BLE name
python scripts/live_translate.py --device "My Myo"
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
for rapid community-level language creation and adoption. Your community
invented a new sign for something that has no word yet. It is yours. Echo makes
sure it stays that way.

The platform scales to every subculture that invents faster than the culture
can document.

---

## Repo Structure

```
echo/
├── emg-asl/        # Live ASL translation: Myo BLE -> LSTM -> Claude
│   ├── scripts/    # live_translate.py, calibrate_quick.py
│   ├── src/        # LSTM model, constants, signal processing
│   └── models/     # Model weights
└── index.html      # Echo landing page
```

---

All glory to God! ✝️❤️

# echo

Real-time ASL translation wristband. Wear a Myo armband, sign — your words appear as natural English and are spoken aloud to your conversation partner.

```
Myo BLE → EMG + IMU → SVM phrase classifier → LLM grammar → English + TTS
```

---

## What it does

- **Live translation** — sign gestures are recognised in real time and converted to English sentences
- **Conversation mode** — two-way: the signing user's words are spoken aloud (ElevenLabs TTS); the hearing partner replies by holding a mic button (Deepgram STT)
- **Teach Echo** — record any new word or phrase in 5 reps and it's immediately added to the model
- **Personalize** — add more reps for existing words from the Settings page to improve accuracy for your signing style
- **Null rejection** — the model stays silent for random arm movements; only real signs produce output

---

## Stack

| Layer | Tech |
|---|---|
| Sensor | Thalmic Myo armband — 8-channel EMG + IMU at 200 Hz over BLE |
| Classifier | SVM with RBF kernel, DTW features, Sakoe-Chiba banded warping |
| Sentence construction | Rule-based ASL→English reordering + Claude Haiku fallback |
| TTS | ElevenLabs (`eleven_turbo_v2_5`) |
| STT | Deepgram Nova-2 (WebSocket streaming) |
| Frontend | Next.js 14 (App Router), Tailwind CSS |
| Backend | Python asyncio WebSocket server (`websockets`) |

---

## Quickstart

### 1. Python backend

```bash
git clone https://github.com/alicej06/echo.git
cd echo
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

Copy `.env` and fill in your keys:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Find your Myo's BLE address, then start the server:
```bash
python scripts/live_translate.py --scan
python scripts/live_translate.py --user alice --ws-port 8765
```

### 2. Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```
NEXT_PUBLIC_DEEPGRAM_API_KEY=...
NEXT_PUBLIC_ELEVENLABS_API_KEY=...
NEXT_PUBLIC_ELEVENLABS_VOICE_ID=...
```

```bash
npm run dev   # http://localhost:3000
```

---

## Training

### First-time setup — record phrase reps

```bash
python scripts/live_translate.py --user alice --train-words
```

Performs 5 reps per phrase interactively. Recordings are saved to `models/user_alice/phrase_recordings.pkl` and persist between sessions.

### Record null / background gestures

```bash
python scripts/live_translate.py --user alice --train-null --train-null-reps 30
```

Vary each rep: arm resting, reaching, pointing, casual wave, transitions between signs. The null class prevents false positives.

### Retrain the model

After recording, retrain from the frontend Train page, or the model is retrained automatically when you finish recording via the UI.

### Evaluate

```bash
python scripts/train_dtw.py --user alice --evaluate
```

Runs leave-one-out cross-validation and prints a confusion matrix.

---

## Vocabulary

Default phrases (9):

| Phrase | ASL hint |
|---|---|
| hello | Wave hand side to side |
| my | Flat hand on chest |
| name | Tap index + middle fingers together |
| echo | Fingerspell E-C-H-O |
| nice to meet you | Flat hand slides off other palm |
| how are you | Bent fingers roll forward, then point |
| thank you | Flat hand from chin forward |
| great | Thumbs up or fist push forward |
| what's your name | WH sign → point at person → name sign |

Add any word or phrase via **Teach Echo** in the app (Settings → Teach, or the Teach tab).

---

## Command reference

### `live_translate.py`

| Flag | Default | Description |
|---|---|---|
| `--user ID` | `default` | User ID for loading/saving models |
| `--device MAC` | auto-discover | Myo BLE MAC address |
| `--ws-port N` | `8765` | WebSocket server port |
| `--train-words` | — | Terminal training mode for phrases |
| `--train-words-reps N` | `5` | Reps per phrase |
| `--train-null` | — | Terminal training mode for null gestures |
| `--train-null-reps N` | `30` | Number of null reps to record |
| `--no-llm` | — | Skip LLM sentence construction |
| `--scan` | — | List nearby BLE devices and exit |
| `--inspect` | — | Stream raw EMG+IMU to terminal |

---

## Repo structure

```
echo/
├── scripts/
│   ├── live_translate.py   # main server — BLE, classifier, WebSocket, LLM
│   ├── train_dtw.py        # SVM training, DTW features, augmentation, evaluation
│   └── train_dyfav.py      # DyFAV static-pose classifier (letter-level)
├── frontend/
│   ├── app/
│   │   ├── home/           # dashboard + recent sessions
│   │   ├── translate/      # live translation view
│   │   ├── conversation/   # two-way ASL ↔ voice chat
│   │   ├── teach/          # teach echo a new gesture
│   │   ├── train/          # record training reps + retrain model
│   │   ├── history/        # past sessions
│   │   └── profile/        # settings + personalization
│   └── hooks/
│       ├── use-myo-ws.ts   # WebSocket client + state
│       ├── use-deepgram.ts # Deepgram STT hook
│       └── use-elevenlabs.ts # ElevenLabs TTS hook
├── models/
│   └── user_<id>/
│       ├── phrase_recordings.pkl  # raw EMG recordings per phrase
│       └── dtw_model.pkl          # trained SVM model
└── requirements.txt
```

---

## What Echo is

Echo is infrastructure for communities to own the language they invent.

Every friend group, every signing community has expressions that exist nowhere in writing. Echo makes that language learnable, permanent, and transferable — starting with ASL, where communities evolve vocabulary faster than any institution can track.

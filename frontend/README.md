# MAIA Frontend

Next.js web interface for the MAIA ASL communication system. Connects to the Python pipeline via WebSocket and displays real-time letter classification from the Myo armband.

## Stack

- Next.js 16 + React 19
- Tailwind CSS v4
- Lucide React icons

## Pages

| Route           | Description                                                                         |
| --------------- | ----------------------------------------------------------------------------------- |
| `/`             | Home: pipeline overview and quick links                                             |
| `/translate`    | Live translation: letter display, confidence, letter stream, Claude sentence output |
| `/conversation` | iMessage-style chat: ASL on left, hold-to-speak mic on right                        |
| `/history`      | Saved sessions from localStorage                                                    |
| `/profile`      | Device info, voice rate, confidence threshold, debounce window                      |

## Running locally

```bash
cd frontend
npm install
npm run dev
# http://localhost:3000
```

## Connecting to the Myo pipeline

Start the Python pipeline with the WebSocket bridge flag:

```bash
cd emg-asl
python scripts/live_translate.py --ws-port 8765
```

Then open `/translate` in the browser, enter `ws://localhost:8765`, and click **Connect**.

The frontend handles three message types from the pipeline:

```json
{ "type": "letter", "letter": "A", "confidence": 0.94 }
{ "type": "sentence", "text": "Hello, how are you?" }
{ "type": "status", "connected": true, "device": "Myo" }
```

## Demo mode

No Myo required. Click **Demo** on the Translate page to simulate letter input cycling through `HELLO WORLD MAIA ASL` with realistic sentence output.

## Session history

Sessions are saved to `localStorage` under `maia_sessions` automatically when you disconnect. View them on the History page.

---

All glory to God! ✝️❤️

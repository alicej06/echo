#!/usr/bin/env python3
"""
live_translate.py — real-time ASL fingerspelling via Myo + DyFAV + LLM.

The script has two modes:
  1. Recognition — stream EMG+IMU from Myo, run DyFAV inference, broadcast via WebSocket
  2. Training    — collect 5 recordings per letter, train a personal DyFAV model

Usage:
    # Find your Myo's BLE name:
    python scripts/live_translate.py --scan

    # Run (training + recognition, WebSocket on port 8765):
    python scripts/live_translate.py --device "Myo" --user alice --ws-port 8765

    # No LLM (letters only):
    python scripts/live_translate.py --device "Myo" --no-llm

WebSocket protocol (JSON):
  Server → Client:
    {"type": "letter",      "letter": "A", "confidence": 0.94,
     "top_k": [["A",0.94],["S",0.03]]}
    {"type": "sentence",    "text": "Hello world"}
    {"type": "status",      "connected": true,  "device": "Myo"}
    {"type": "train_ack",   "letter": "a", "count": 3, "needed": 5}
    {"type": "model_ready", "user_id": "alice"}
    {"type": "error",       "message": "..."}

  Client → Server:
    {"type": "train_record", "letter": "a"}   -- collect one training recording
    {"type": "train_model"}                    -- trigger model training
    {"type": "correction",   "letter": "b"}   -- user correction feedback

LLM backends (priority order):
    1. Claude Haiku   — set ANTHROPIC_API_KEY
    2. Ollama cloud   — set OLLAMA_API_KEY
    3. Ollama local   — install Ollama + ollama pull llama3.2
    4. --no-llm       — letters only
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preprocess import extract_510_features
from scripts.train_dyfav import (
    ALL_LETTERS, TOP_K, predict, predict_topk, FUZZY_THRESHOLD_DEFAULT,
    train_from_recordings, train_for_user, load_user_model,
)
SAMPLE_RATE = 200   # Hz (Myo Armband EMG)
DEBOUNCE_MS = 300

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR         = ROOT / "models"
RECORDINGS_DIR     = ROOT / "data" / "recordings"   # temp storage for training data
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

COLLECT_SAMPLES    = 51     # samples per recording (matches training data)
INFER_HOP_SAMPLES  = 20     # run inference every N new EMG samples
CONFIDENCE_THRESH  = 0.10   # minimum absolute DyFAV score for top letter
MARGIN_THRESH      = 0.03   # top score must beat 2nd by at least this much
STABLE_FRAMES      = 6      # same letter must hold for this many consecutive frames before emitting
LLM_PAUSE_S        = 1.8
LLM_MIN_LETTERS    = 2
TRAIN_REPS_NEEDED  = 5

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_API_KEY     = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_CLOUD_URL   = "https://api.ollama.com/api/chat"
OLLAMA_CLOUD_MODEL = os.environ.get("OLLAMA_CLOUD_MODEL", "gemma3:4b")
OLLAMA_LOCAL_URL   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_LOCAL_MODEL = os.environ.get("OLLAMA_MODEL",    "llama3.2")

_NYQ = SAMPLE_RATE / 2.0
_SOS_EMG = butter(4, [20.0 / _NYQ, 95.0 / _NYQ], btype="band", output="sos")

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

R      = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
VIOLET = "\033[95m"
DIM    = "\033[2m"


def clr(text: str, code: str) -> str:
    return f"{code}{text}{R}"

# ---------------------------------------------------------------------------
# WebSocket broadcast server
# ---------------------------------------------------------------------------

_ws_clients: set = set()


async def _ws_broadcast(payload: dict) -> None:
    if not _ws_clients:
        return
    msg = json.dumps(payload)
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# _ws_message_handler is set by _make_session so the WS server can route
# incoming client messages to the session logic
_ws_message_handler = None


async def _ws_server_handler(websocket) -> None:
    _ws_clients.add(websocket)
    try:
        async for raw in websocket:
            if _ws_message_handler:
                try:
                    msg = json.loads(raw)
                    await _ws_message_handler(msg)
                except Exception:
                    pass
    finally:
        _ws_clients.discard(websocket)


async def _start_ws_server(port: int) -> None:
    try:
        import websockets  # type: ignore[import]
    except ImportError:
        print(f"  {clr('[ws]', YELLOW)}  websockets not installed — pip install websockets")
        return
    server = await websockets.serve(_ws_server_handler, "0.0.0.0", port)
    print(f"  {clr('ws', CYAN)}  broadcast server → ws://localhost:{port}")
    await server.wait_closed()

# ---------------------------------------------------------------------------
# BLE scanner
# ---------------------------------------------------------------------------

async def scan_devices() -> None:
    from bleak import BleakScanner
    print(clr("\n  Scanning for BLE devices (8 seconds)...\n", CYAN))
    devices = await BleakScanner.discover(timeout=8.0)
    named = [(d.address, d.name) for d in devices if d.name]
    if not named:
        print(clr("  No named BLE devices found. Is Bluetooth on?", RED))
        return
    print(f"  {'Address':<40}  Name")
    print(f"  {'-'*40}  {'-'*30}")
    for addr, name in sorted(named, key=lambda x: x[1] or ""):
        arrow = clr("  <-- likely your Myo", GREEN) if "myo" in (name or "").lower() else ""
        print(f"  {addr:<40}  {name}{arrow}")

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

LLM_SYSTEM = (
    "You are an ASL-to-English interpreter. "
    "The user sends fingerspelled letters (possibly with recognition errors). "
    "Reconstruct them into natural conversational English. "
    "Handle substitution errors gracefully. "
    "Output only the reconstructed English. No quotes, no explanation."
)


def _try_anthropic(text: str) -> str | None:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        print(f"\n  {clr('[anthropic]', YELLOW)} {exc}")
        return None


def _try_ollama_cloud(text: str) -> str | None:
    if not OLLAMA_API_KEY:
        return None
    try:
        payload = json.dumps({
            "model": OLLAMA_CLOUD_MODEL, "stream": False,
            "messages": [{"role": "system", "content": LLM_SYSTEM},
                         {"role": "user",   "content": text}],
        }).encode()
        req = urllib.request.Request(
            OLLAMA_CLOUD_URL, data=payload,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["message"]["content"].strip()
    except Exception:
        return None


def _try_ollama_local(text: str) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=OLLAMA_LOCAL_URL)
        resp = client.chat.completions.create(
            model=OLLAMA_LOCAL_MODEL, max_tokens=128,
            messages=[{"role": "system", "content": LLM_SYSTEM},
                      {"role": "user",   "content": text}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


async def llm_translate(letters: list[str]) -> str:
    text = "".join(letters)
    return (
        _try_anthropic(text)
        or _try_ollama_cloud(text)
        or _try_ollama_local(text)
        or "[no LLM — set ANTHROPIC_API_KEY or install Ollama]"
    )

# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def render(letter: str, score: float, top_k: list, letter_stream: list[str],
           sentence: str, mode: str) -> None:
    bar_len = 20
    filled = int(min(score, 1.0) * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    color = GREEN if score >= 0.5 else (YELLOW if score >= 0.3 else RED)
    stream_str = " ".join(letter_stream[-20:]) or clr("(waiting)", DIM)
    top_str = "  ".join(f"{l}:{s:.2f}" for l, s in top_k[:3]) if top_k else ""
    mode_str = clr(f"[{mode}]", CYAN)
    sentence_str = clr(sentence, BOLD + VIOLET) if sentence else clr("(pause to translate)", DIM)
    print(
        f"\033[4A"
        f"\r  {mode_str}  {clr(letter, BOLD)}  [{clr(bar, color)}] {score:.2f}\033[K\n"
        f"\r  {clr('top', DIM)}: {top_str}\033[K\n"
        f"\r  {clr('letters', DIM)}: {stream_str}\033[K\n"
        f"\r  {clr('echo', CYAN)}: {sentence_str}\033[K",
        end="", flush=True,
    )

# ---------------------------------------------------------------------------
# Myo session — EMG + IMU, DyFAV inference, training support
# ---------------------------------------------------------------------------

def _make_session(
    user_id: str,
    device_mac: str,
    use_llm: bool,
    ws_port: int = 0,
):
    global _ws_message_handler

    # Load user model (may be None if not yet trained)
    dyfav_model = [load_user_model(user_id)]
    if dyfav_model[0]:
        print(f"  {clr('model', CYAN)}  loaded user model for '{user_id}'")
    else:
        print(f"  {clr('model', YELLOW)}  no model for user '{user_id}' — use training mode first")

    # Training state
    # recordings[letter] = list of (n_samples, 17) arrays
    recordings: dict[str, list[np.ndarray]] = {}
    collecting         = [False]
    collect_letter     = [None]
    collect_buf: list  = []   # rows of 17 values being collected

    # Inference state
    # emg_buf: rolling deque of 8-channel EMG rows (after bandpass)
    # imu_buf: latest IMU row [qw,qx,qy, gx,gy,gz, ax,ay,az]
    emg_buf   = collections.deque(maxlen=COLLECT_SAMPLES * 4)
    imu_state = [np.zeros(9, dtype=np.float32)]  # last known IMU
    new_emg   = [0]

    letter_stream:  list[str] = []
    pending_letters: list[str] = []
    sentence        = [""]
    last_emit_ts    = [0.0]
    last_letter_ts  = [0.0]
    llm_running     = [False]
    mode            = ["recognition"]  # "recognition" | "training"

    # Gesture stability state
    stable_letter   = [None]   # letter currently being held
    stable_count    = [0]      # how many consecutive frames it's been top
    last_emitted    = [None]   # last letter we actually emitted (require change to re-emit)

    def _build_sync_array() -> np.ndarray:
        """Combine most recent COLLECT_SAMPLES EMG rows with latest IMU.
        Returns (COLLECT_SAMPLES, 17) float32."""
        emg = np.array(list(emg_buf)[-COLLECT_SAMPLES:], dtype=np.float32)
        imu = np.tile(imu_state[0], (len(emg), 1))
        return np.column_stack([emg, imu])

    def _infer() -> None:
        if dyfav_model[0] is None:
            return
        if len(emg_buf) < COLLECT_SAMPLES:
            return

        raw = _build_sync_array()
        try:
            feat = extract_510_features(raw)
            pred_idx, scores = predict(dyfav_model[0], feat, FUZZY_THRESHOLD_DEFAULT)
            letter = ALL_LETTERS[pred_idx].upper()
            score  = scores[pred_idx]
            top_k  = predict_topk(dyfav_model[0], feat, k=5)
        except Exception:
            return

        # Compute margin: gap between 1st and 2nd place scores
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else score

        # Stability tracking: count consecutive frames with the same top letter
        if letter == stable_letter[0]:
            stable_count[0] += 1
        else:
            stable_letter[0] = letter
            stable_count[0]  = 1

        # Emit only when:
        #   1. Score beats absolute threshold
        #   2. Margin over 2nd place is clear
        #   3. Letter has been stable for enough frames
        #   4. Letter is different from last emitted (no repeat spam)
        gesture_clear = (
            score  >= CONFIDENCE_THRESH
            and margin >= MARGIN_THRESH
            and stable_count[0] >= STABLE_FRAMES
            and letter != last_emitted[0]
        )

        if gesture_clear:
            letter_stream.append(letter)
            pending_letters.append(letter)
            last_emit_ts[0]   = time.monotonic()
            last_letter_ts[0] = time.monotonic()
            last_emitted[0]   = letter
            stable_count[0]   = 0  # reset so same letter requires re-stabilisation
            if ws_port:
                asyncio.create_task(_ws_broadcast({
                    "type": "letter",
                    "letter": letter,
                    "confidence": round(score, 4),
                    "top_k": top_k,
                }))

        render(letter, score, top_k, letter_stream, sentence[0], mode[0])

    async def _handle_ws_message(msg: dict) -> None:
        """Handle messages sent from the frontend to the server."""
        mtype = msg.get("type", "")

        if mtype == "train_record":
            # Client wants to record one training sample for a letter
            letter = str(msg.get("letter", "")).lower()
            if letter not in ALL_LETTERS:
                await _ws_broadcast({"type": "error", "message": f"Invalid letter: {letter}"})
                return
            if collecting[0]:
                await _ws_broadcast({"type": "error", "message": "Already recording"})
                return

            collecting[0]    = True
            collect_letter[0] = letter
            collect_buf.clear()
            mode[0] = f"collecting {letter.upper()}"
            print(f"\n  {clr('train', CYAN)}  collecting '{letter.upper()}'...")

        elif mtype == "train_model":
            # Client wants to train the model with collected recordings
            missing = [l for l in ALL_LETTERS if len(recordings.get(l, [])) < TRAIN_REPS_NEEDED]
            if missing:
                await _ws_broadcast({
                    "type": "error",
                    "message": f"Need {TRAIN_REPS_NEEDED} recs each — missing: {[l.upper() for l in missing[:5]]}...",
                })
                return

            print(f"\n  {clr('train', CYAN)}  training DyFAV for user '{user_id}'...")
            model = train_from_recordings(recordings)
            model["user_id"] = user_id
            dyfav_model[0] = model
            # Save to disk
            import joblib
            user_dir = MODELS_DIR / f"user_{user_id}"
            user_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, user_dir / "dyfav_model.pkl")
            recordings.clear()
            mode[0] = "recognition"
            print(f"  {clr('train', GREEN)}  model saved → models/user_{user_id}/dyfav_model.pkl")
            await _ws_broadcast({"type": "model_ready", "user_id": user_id})

        elif mtype == "correction":
            # User corrected a misclassification — store for future retraining
            letter = str(msg.get("letter", "")).lower()
            print(f"\n  {clr('correction', YELLOW)}  user says: {letter.upper()}")

    _ws_message_handler = _handle_ws_message

    from myo import MyoClient
    from myo.types import (
        EMGData, IMUData, EMGMode, IMUMode, ClassifierMode,
        AggregatedData, ClassifierEvent, EMGDataSingle, FVData, MotionEvent,
    )

    class EchoMyo(MyoClient):

        async def on_emg_data(self, emg: EMGData) -> None:
            for sample in (emg.sample1, emg.sample2):
                row = np.array(list(sample), dtype=np.float32)
                # Bandpass filter each sample via running state not possible sample-by-sample,
                # so we just append and filter the window at inference time.
                emg_buf.append(row)

                if collecting[0]:
                    # Combine EMG with latest IMU to get 17-dim row
                    full_row = np.concatenate([row, imu_state[0]])
                    collect_buf.append(full_row)

                    if len(collect_buf) >= COLLECT_SAMPLES:
                        # Recording complete
                        raw = np.array(collect_buf[:COLLECT_SAMPLES], dtype=np.float32)
                        letter = collect_letter[0]
                        recordings.setdefault(letter, []).append(raw)
                        count = len(recordings[letter])
                        collecting[0] = False
                        mode[0] = "recognition"
                        print(f"  {clr('train', CYAN)}  '{letter.upper()}' recorded ({count}/{TRAIN_REPS_NEEDED})")
                        if ws_port:
                            asyncio.create_task(_ws_broadcast({
                                "type": "train_ack",
                                "letter": letter,
                                "count": count,
                                "needed": TRAIN_REPS_NEEDED,
                            }))

            new_emg[0] += 2
            if not collecting[0] and new_emg[0] >= INFER_HOP_SAMPLES:
                new_emg[0] = 0
                _infer()

        async def on_imu_data(self, imu: IMUData) -> None:
            # orientation: object with .w .x .y .z
            # gyroscope, accelerometer: plain lists [x, y, z]
            try:
                o = imu.orientation
                g = imu.gyroscope     # list
                a = imu.accelerometer # list
                imu_state[0] = np.array([
                    o.w, o.x, o.y,
                    g[0], g[1], g[2],
                    a[0], a[1], a[2],
                ], dtype=np.float32)
            except Exception:
                pass

        async def on_emg_data_aggregated(self, eds: EMGDataSingle) -> None:
            pass

        async def on_aggregated_data(self, ad: AggregatedData) -> None:
            pass

        async def on_fv_data(self, fvd: FVData) -> None:
            pass

        async def on_motion_event(self, me: MotionEvent) -> None:
            pass

        async def on_classifier_event(self, ce: ClassifierEvent) -> None:
            pass

    async def _llm_loop() -> None:
        while True:
            await asyncio.sleep(0.2)
            if not use_llm or llm_running[0] or not pending_letters:
                continue
            if (time.monotonic() - last_letter_ts[0]) < LLM_PAUSE_S:
                continue
            if len(pending_letters) < LLM_MIN_LETTERS:
                continue
            to_translate = list(pending_letters)
            pending_letters.clear()
            llm_running[0] = True
            sentence[0] = "..."
            sentence[0] = await llm_translate(to_translate)
            llm_running[0] = False
            if ws_port:
                await _ws_broadcast({"type": "sentence", "text": sentence[0]})

    async def run() -> None:
        if device_mac:
            print(f"  {clr('ble', CYAN)}  connecting to MAC {device_mac}...")
        else:
            print(f"  {clr('ble', CYAN)}  scanning for Myo (auto-discover by service UUID)...")
        client = await EchoMyo.with_device(mac=device_mac or None)
        print(f"  {clr('ble', CYAN)}  connected")

        if ws_port:
            await _ws_broadcast({"type": "status", "connected": True, "device": device_mac or "Myo"})

        await client.setup(
            emg_mode=EMGMode.SEND_EMG,
            imu_mode=IMUMode.SEND_ALL,          # enable IMU for DyFAV
            classifier_mode=ClassifierMode.DISABLED,
        )
        print(f"  {clr('myo', CYAN)}  EMG + IMU streaming enabled")
        print()
        print(f"  {clr('Echo running', BOLD + GREEN)}"
              f"  |  user: {clr(user_id, CYAN)}"
              f"  |  model: {clr('loaded' if dyfav_model[0] else 'NONE — train first', YELLOW if not dyfav_model[0] else GREEN)}")
        if use_llm:
            llm_name = "Claude Haiku" if ANTHROPIC_API_KEY else f"Ollama ({OLLAMA_LOCAL_MODEL})"
            print(f"  {clr('LLM:', DIM)} {clr(llm_name, CYAN)}")
        print(); print(); print(); print()

        llm_task = asyncio.create_task(_llm_loop())
        try:
            await client.start()
            while True:
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            llm_task.cancel()
            await client.stop()
            await client.disconnect()
            if ws_port:
                await _ws_broadcast({"type": "status", "connected": False, "device": device_mac or "Myo"})
            print(f"\n\n  {clr('transcript', DIM)}: {' '.join(letter_stream)}")
            print(f"  {clr('last echo', VIOLET)}: {sentence[0]}")

    return run

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    print()
    print(clr("  Echo  |  ASL Fingerspelling  |  DyFAV", BOLD))
    print(clr("  Myo BLE → EMG+IMU → DyFAV → letters → LLM → English", DIM))
    print()

    if args.scan:
        await scan_devices()
        return

    ws_port = args.ws_port or 0
    run = _make_session(
        user_id=args.user,
        device_mac=args.device,
        use_llm=not args.no_llm,
        ws_port=ws_port,
    )

    tasks = [asyncio.create_task(run())]
    if ws_port:
        tasks.append(asyncio.create_task(_start_ws_server(ws_port)))
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo — live ASL translation from Myo armband")
    parser.add_argument("--device",   default="",       help="Myo MAC address (optional; auto-discovers if omitted)")
    parser.add_argument("--user",     default="default", help="User ID for model loading/saving")
    parser.add_argument("--scan",     action="store_true", help="Scan for BLE devices and exit")
    parser.add_argument("--no-llm",  action="store_true", help="Skip LLM sentence reconstruction")
    parser.add_argument("--ws-port", type=int, default=8765, dest="ws_port",
                        help="WebSocket server port (default: 8765)")
    asyncio.run(main(parser.parse_args()))

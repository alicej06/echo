#!/usr/bin/env python3
"""
live_translate.py — real-time ASL -> natural English via Myo + LLM.

Usage (from project root):
    source .venv/bin/activate
    python3 scripts/live_translate.py --device "My Myo"

    # Find your Myo's BLE name first:
    python3 scripts/live_translate.py --scan

    # Skip LLM (letters only):
    python3 scripts/live_translate.py --no-llm

    # Use a personal calibrated model:
    python3 scripts/live_translate.py --model models/calibrated/me/model.pt

LLM backends (automatic priority order):
    1. Claude Haiku      — set ANTHROPIC_API_KEY
    2. Ollama cloud      — set OLLAMA_API_KEY (free, uses gemma3:4b)
    3. Ollama local      — install Ollama + ollama pull llama3.2
    4. --no-llm          — letters only
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    ASL_CLASSES, CONFIDENCE_THRESHOLD, DEBOUNCE_MS, HOP_SAMPLES,
    LSTM_DROPOUT, LSTM_HIDDEN, LSTM_LAYERS,
    N_CHANNELS, N_CLASSES, SAMPLE_RATE, WINDOW_SAMPLES,
)
from src.models.lstm_classifier import LSTMClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH  = ROOT / "models" / "asl_emg_classifier.pt"
BUFFER_MAXLEN       = SAMPLE_RATE * 4
LLM_PAUSE_S         = 1.8
LLM_MIN_LETTERS     = 2

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_API_KEY      = os.environ.get("OLLAMA_API_KEY", "74dac44848a745fc8dd672fa57e7fb47.QH8v_5DvIscL1voqyrV48SAw")
OLLAMA_CLOUD_URL    = "https://api.ollama.com/api/chat"
OLLAMA_CLOUD_MODEL  = os.environ.get("OLLAMA_CLOUD_MODEL", "gemma3:4b")
OLLAMA_LOCAL_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_LOCAL_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3.2")

_NYQ = SAMPLE_RATE / 2.0
_SOS = butter(4, [20.0 / _NYQ, 95.0 / _NYQ], btype="band", output="sos")

# ---------------------------------------------------------------------------
# WebSocket broadcast server (optional, --ws-port)
# Broadcasts JSON events to connected browser clients:
#   {"type": "letter", "letter": "A", "confidence": 0.94}
#   {"type": "sentence", "text": "Hello world"}
#   {"type": "status", "connected": true, "device": "My Myo"}
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


async def _ws_server_handler(websocket) -> None:  # type: ignore[type-arg]
    _ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _ws_clients.discard(websocket)


async def _start_ws_server(port: int, clr_fn, CYAN: str, YELLOW: str) -> None:
    try:
        import websockets  # type: ignore[import]
    except ImportError:
        print(f"  {clr_fn('[ws]', YELLOW)}  websockets not installed — run: pip install websockets")
        return
    server = await websockets.serve(_ws_server_handler, "localhost", port)
    print(f"  {clr_fn('ws', CYAN)}  broadcast server on ws://localhost:{port}")
    await server.wait_closed()

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
    print(f"\n  Run with: python3 scripts/live_translate.py --device \"<name>\"")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(path: Path) -> LSTMClassifier:
    model = LSTMClassifier(
        n_channels=N_CHANNELS, n_classes=N_CLASSES,
        hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT,
    )
    if path.exists():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            print(f"  {clr('model', CYAN)}  loaded {path.name}")
        except Exception as exc:
            print(f"  {clr('model', YELLOW)}  could not load ({exc}) — using random init")
    else:
        print(f"  {clr('model', YELLOW)}  {path.name} not found — random init")
        print(f"           run: python3 scripts/calibrate_quick.py --user me")
    model.eval()
    return model


def predict(model: LSTMClassifier, window: np.ndarray) -> tuple[str, float]:
    x = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_proba(x)[0]
    idx = int(probs.argmax())
    return ASL_CLASSES[idx], float(probs[idx])

# ---------------------------------------------------------------------------
# LLM backends — priority: Anthropic > Ollama cloud > Ollama local
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
        print(f"\n  {clr('[anthropic]', YELLOW)} {exc} — trying Ollama cloud")
        return None


def _try_ollama_cloud(text: str) -> str | None:
    if not OLLAMA_API_KEY:
        return None
    try:
        payload = json.dumps({
            "model": OLLAMA_CLOUD_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user",   "content": text},
            ],
        }).encode()
        req = urllib.request.Request(
            OLLAMA_CLOUD_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return result["message"]["content"].strip()
    except Exception as exc:
        print(f"\n  {clr('[ollama cloud]', YELLOW)} {exc} — trying local Ollama")
        return None


def _try_ollama_local(text: str) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=OLLAMA_LOCAL_URL)
        resp = client.chat.completions.create(
            model=OLLAMA_LOCAL_MODEL,
            max_tokens=128,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user",   "content": text},
            ],
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
        or "[no LLM available]"
    )


def active_llm_label() -> str:
    if ANTHROPIC_API_KEY:
        return clr("Claude Haiku", CYAN)
    if OLLAMA_API_KEY:
        return clr(f"Ollama cloud ({OLLAMA_CLOUD_MODEL})", CYAN)
    return clr(f"Ollama local ({OLLAMA_LOCAL_MODEL})", YELLOW)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def render(letter: str, conf: float, letter_stream: list[str], sentence: str) -> None:
    bar_len = 20
    filled = int(conf * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    color = GREEN if conf >= 0.85 else (YELLOW if conf >= 0.65 else RED)
    stream_str = " ".join(letter_stream[-20:]) or clr("(waiting for input)", DIM)
    sentence_str = clr(sentence, BOLD + VIOLET) if sentence else clr("(will appear after you pause)", DIM)
    print(
        f"\033[3A"
        f"\r  {clr(letter, BOLD)}  [{clr(bar, color)}] {clr(f'{conf:.0%}', color)}\033[K\n"
        f"\r  {clr('letters', DIM)}: {stream_str}\033[K\n"
        f"\r  {clr('echo', CYAN)}: {sentence_str}\033[K",
        end="", flush=True,
    )

# ---------------------------------------------------------------------------
# Myo session — uses dl-myo for clean BLE management
# ---------------------------------------------------------------------------

def _make_session(model: LSTMClassifier, device_name: str, use_llm: bool, ws_port: int = 0):
    """Build and return a MyoClient subclass instance ready to run."""
    from myo import MyoClient
    from myo.types import (
        AggregatedData, ClassifierEvent, EMGData, EMGDataSingle,
        EMGMode, IMUMode, ClassifierMode, FVData, IMUData, MotionEvent,
    )

    buf: collections.deque = collections.deque(maxlen=BUFFER_MAXLEN)
    new_since_last = [0]
    last_emit_ts   = [0.0]
    last_letter_ts = [0.0]
    letter_stream: list[str] = []
    pending_letters: list[str] = []
    sentence       = [""]
    llm_running    = [False]

    def _infer() -> None:
        if len(buf) < WINDOW_SAMPLES:
            return
        window = np.array(list(buf)[-WINDOW_SAMPLES:], dtype=np.float32) / 128.0
        window = sosfilt(_SOS, window, axis=0).astype(np.float32)
        letter, conf = predict(model, window)
        now = time.monotonic()
        if conf >= CONFIDENCE_THRESHOLD and (now - last_emit_ts[0]) >= DEBOUNCE_MS / 1000.0:
            letter_stream.append(letter)
            pending_letters.append(letter)
            last_emit_ts[0] = now
            last_letter_ts[0] = now
            if ws_port:
                asyncio.create_task(_ws_broadcast({"type": "letter", "letter": letter, "confidence": round(float(conf), 4)}))
        render(letter, conf, letter_stream, sentence[0])

    class EchoMyo(MyoClient):
        async def on_emg_data(self, emg: EMGData) -> None:
            # Two samples per packet, 8 channels each — identical to raw BLE
            buf.append(list(emg.sample1))
            buf.append(list(emg.sample2))
            new_since_last[0] += 2
            if new_since_last[0] >= HOP_SAMPLES:
                new_since_last[0] = 0
                _infer()

        async def on_emg_data_aggregated(self, eds: EMGDataSingle) -> None:
            pass

        async def on_aggregated_data(self, ad: AggregatedData) -> None:
            pass

        async def on_fv_data(self, fvd: FVData) -> None:
            pass

        async def on_imu_data(self, imu: IMUData) -> None:
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
            sentence[0] = clr("...", DIM)
            sentence[0] = await llm_translate(to_translate)
            llm_running[0] = False
            if ws_port:
                await _ws_broadcast({"type": "sentence", "text": sentence[0]})

    async def run() -> None:
        print(f"  {clr('ble', CYAN)}  scanning for '{device_name}'...")

        client = await EchoMyo.with_device(name=device_name)
        print(f"  {clr('ble', CYAN)}  connected")
        if ws_port:
            await _ws_broadcast({"type": "status", "connected": True, "device": device_name})

        await client.setup(
            emg_mode=EMGMode.SEND_EMG,
            imu_mode=IMUMode.NONE,
            classifier_mode=ClassifierMode.DISABLED,
        )
        print(f"  {clr('myo', CYAN)}  EMG streaming enabled")

        print()
        print(f"  {clr('live translation running', BOLD + GREEN)}")
        if use_llm:
            print(f"  {clr('LLM:', DIM)} {active_llm_label()} — pause signing for natural English")
        else:
            print(f"  {clr('letters only mode', DIM)}")
        print(); print(); print()

        llm_task = asyncio.create_task(_llm_loop())
        try:
            await client.start()
            # keep running until Ctrl-C
            while True:
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            llm_task.cancel()
            await client.stop()
            await client.disconnect()
            print(f"\n\n  {clr('transcript', DIM)}: {' '.join(letter_stream)}")
            print(f"  {clr('last echo', VIOLET)}: {sentence[0]}")

    return run

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    print()
    print(clr("  Echo  |  Live ASL Translation", BOLD))
    print(clr("  Myo BLE -> LSTM -> letters -> LLM -> English", DIM))
    print()

    if args.scan:
        await scan_devices()
        return

    model = load_model(Path(args.model))
    ws_port = getattr(args, "ws_port", 0) or 0
    run = _make_session(model, device_name=args.device, use_llm=not args.no_llm, ws_port=ws_port)
    tasks = [asyncio.create_task(run())]
    if ws_port:
        tasks.append(asyncio.create_task(_start_ws_server(ws_port, clr, CYAN, YELLOW)))
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo — live ASL translation from Myo armband")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="Myo", help="BLE name of your Myo (default: Myo)")
    parser.add_argument("--scan", action="store_true", help="List nearby BLE devices and exit")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM sentence reconstruction")
    parser.add_argument("--ws-port", type=int, default=0, dest="ws_port",
                        help="Start a WebSocket broadcast server on this port for the web frontend (e.g. 8765)")
    asyncio.run(main(parser.parse_args()))

#!/usr/bin/env python3
"""
live_translate.py — real-time ASL -> natural English via Myo + LLM.

Usage (from project root):
    python scripts/live_translate.py

    # Find your Myo's BLE name first:
    python scripts/live_translate.py --scan

    # If your Myo shows as something other than "Myo":
    python scripts/live_translate.py --device "My Myo"

    # Skip LLM (no API key needed):
    python scripts/live_translate.py --no-llm

    # Use a personal calibrated model:
    python scripts/live_translate.py --model models/calibrated/me/model.pt

Requirements:
    pip install -r requirements-live.txt

LLM backend (automatic fallback):
    1. Claude Haiku  — set ANTHROPIC_API_KEY env var (best quality)
    2. Ollama local  — install Ollama + run: ollama pull llama3.2
       Falls back automatically when ANTHROPIC_API_KEY is unset or errors.
    3. --no-llm      — letters only, no sentence reconstruction
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    ASL_CLASSES, BLE_CONTROL_UUID, BLE_EMG_CHAR_UUIDS,
    CONFIDENCE_THRESHOLD, DEBOUNCE_MS, HOP_SAMPLES,
    LSTM_DROPOUT, LSTM_HIDDEN, LSTM_LAYERS, MYO_ENABLE_EMG_CMD,
    N_CHANNELS, N_CLASSES, SAMPLE_RATE, WINDOW_SAMPLES,
)
from src.models.lstm_classifier import LSTMClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = ROOT / "models" / "asl_emg_classifier.pt"
BUFFER_MAXLEN      = SAMPLE_RATE * 4
LLM_PAUSE_S        = 1.8
LLM_MIN_LETTERS    = 2
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "llama3.2")

_NYQ = SAMPLE_RATE / 2.0
_SOS = butter(4, [20.0 / _NYQ, 95.0 / _NYQ], btype="band", output="sos")

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
# BLE scanner utility
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
    print(f"\n  Run with: python scripts/live_translate.py --device \"<name>\"")

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
        print(f"           run: python scripts/calibrate_quick.py --user me")
    model.eval()
    return model


def predict(model: LSTMClassifier, window: np.ndarray) -> tuple[str, float]:
    x = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        probs = model.predict_proba(x)[0]
    idx = int(probs.argmax())
    return ASL_CLASSES[idx], float(probs[idx])

# ---------------------------------------------------------------------------
# LLM translation — Claude Haiku with automatic Ollama fallback
# ---------------------------------------------------------------------------

LLM_SYSTEM = (
    "You are an ASL-to-English interpreter. "
    "The user sends fingerspelled letters (possibly with recognition errors). "
    "Reconstruct them into natural conversational English. "
    "Handle substitution errors gracefully. "
    "Output only the reconstructed English. No quotes, no explanation."
)


def _try_anthropic(letters_str: str) -> str | None:
    """Attempt Claude Haiku. Returns text on success, None on any failure."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": letters_str}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        print(f"\n  {clr('[anthropic error]', YELLOW)} {exc} — trying Ollama fallback")
        return None


def _try_ollama(letters_str: str) -> str | None:
    """Attempt Ollama via OpenAI-compat endpoint. Returns text or None."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=128,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user",   "content": letters_str},
            ],
        )
        result = resp.choices[0].message.content.strip()
        print(f"\n  {clr('[ollama]', DIM)} used {OLLAMA_MODEL} for reconstruction")
        return result
    except Exception as exc:
        return f"[LLM error: {exc}]"


async def llm_translate(letters: list[str]) -> str:
    letters_str = "".join(letters)
    # Try Anthropic first, fall back to Ollama automatically
    result = _try_anthropic(letters_str)
    if result is not None:
        return result
    return _try_ollama(letters_str) or "[set ANTHROPIC_API_KEY or install Ollama to enable reconstruction]"

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
# BLE session
# ---------------------------------------------------------------------------

class MyoSession:
    def __init__(self, model: LSTMClassifier, device_name: str, use_llm: bool) -> None:
        self.model = model
        self.device_name = device_name
        self.use_llm = use_llm
        self._buf: collections.deque[list[int]] = collections.deque(maxlen=BUFFER_MAXLEN)
        self._new_since_last = 0
        self._last_emit_ts = 0.0
        self._last_letter_ts = 0.0
        self._letter_stream: list[str] = []
        self._pending_letters: list[str] = []
        self._sentence = ""
        self._llm_running = False

    def _on_emg(self, _handle: int, data: bytearray) -> None:
        if len(data) < 16:
            return
        for i in range(2):
            self._buf.append(list(struct.unpack_from("8b", data, i * 8)))
            self._new_since_last += 1

    def _infer(self) -> None:
        if len(self._buf) < WINDOW_SAMPLES:
            return
        window = np.array(list(self._buf)[-WINDOW_SAMPLES:], dtype=np.float32) / 128.0
        window = sosfilt(_SOS, window, axis=0).astype(np.float32)
        letter, conf = predict(self.model, window)
        now = time.monotonic()
        if conf >= CONFIDENCE_THRESHOLD and (now - self._last_emit_ts) >= DEBOUNCE_MS / 1000.0:
            self._letter_stream.append(letter)
            self._pending_letters.append(letter)
            self._last_emit_ts = now
            self._last_letter_ts = now
        render(letter, conf, self._letter_stream, self._sentence)

    async def _llm_loop(self) -> None:
        while True:
            await asyncio.sleep(0.2)
            if not self.use_llm or self._llm_running or not self._pending_letters:
                continue
            if (time.monotonic() - self._last_letter_ts) < LLM_PAUSE_S:
                continue
            if len(self._pending_letters) < LLM_MIN_LETTERS:
                continue
            to_translate = list(self._pending_letters)
            self._pending_letters.clear()
            self._llm_running = True
            self._sentence = clr("...", DIM)
            self._sentence = await llm_translate(to_translate)
            self._llm_running = False

    async def run(self) -> None:
        from bleak import BleakClient, BleakScanner

        print(f"  {clr('ble', CYAN)}  scanning for '{self.device_name}'...")
        device = await BleakScanner.find_device_by_name(self.device_name, timeout=10.0)
        if device is None:
            print(
                f"\n  {clr('not found', RED)}  '{self.device_name}' not visible.\n"
                f"\n  Find your Myo's name:\n"
                f"  {clr('python scripts/live_translate.py --scan', CYAN)}\n"
                f"\n  Then re-run with:\n"
                f"  {clr('python scripts/live_translate.py --device \"<name>\"', CYAN)}\n"
            )
            return

        print(f"  {clr('ble', CYAN)}  found {device.name} ({device.address})")

        async with BleakClient(device, timeout=15.0) as client:
            print(f"  {clr('ble', CYAN)}  connected")
            try:
                await client.write_gatt_char(BLE_CONTROL_UUID, MYO_ENABLE_EMG_CMD, response=True)
                print(f"  {clr('myo', CYAN)}  raw EMG streaming enabled")
            except Exception as exc:
                print(f"  {clr('myo', YELLOW)}  enable cmd failed ({exc}), continuing")

            for uuid in BLE_EMG_CHAR_UUIDS:
                try:
                    await client.start_notify(uuid, self._on_emg)
                except Exception:
                    pass

            # Show which LLM backend is active
            print()
            print(f"  {clr('live translation running', BOLD + GREEN)}")
            if not self.use_llm:
                print(f"  {clr('letters only mode', DIM)}")
            elif ANTHROPIC_API_KEY:
                print(f"  {clr('LLM: Claude Haiku', CYAN)} — pause signing for natural English")
            else:
                print(f"  {clr('LLM: Ollama fallback', YELLOW)} ({OLLAMA_MODEL}) — ANTHROPIC_API_KEY not set")
            print(); print(); print()

            llm_task = asyncio.create_task(self._llm_loop())
            hop_s = HOP_SAMPLES / SAMPLE_RATE

            try:
                while True:
                    await asyncio.sleep(hop_s)
                    if self._new_since_last >= HOP_SAMPLES:
                        self._new_since_last = 0
                        self._infer()
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            finally:
                llm_task.cancel()
                for uuid in BLE_EMG_CHAR_UUIDS:
                    try:
                        await client.stop_notify(uuid)
                    except Exception:
                        pass
                print(f"\n\n  {clr('transcript', DIM)}: {' '.join(self._letter_stream)}")
                print(f"  {clr('last echo', VIOLET)}: {self._sentence}")

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
    session = MyoSession(model, device_name=args.device, use_llm=not args.no_llm)
    try:
        await session.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo — live ASL translation from Myo armband")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="Myo", help="BLE name of your Myo (default: Myo)")
    parser.add_argument("--scan", action="store_true", help="List nearby BLE devices and exit")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM sentence reconstruction")
    asyncio.run(main(parser.parse_args()))

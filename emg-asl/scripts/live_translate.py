#!/usr/bin/env python3
"""
live_translate.py — real-time ASL -> natural English via Myo + LLM.

Usage (from project root):
    python scripts/live_translate.py

    # Skip LLM (letters only, no API key needed):
    python scripts/live_translate.py --no-llm

    # Use a personal calibrated model:
    python scripts/live_translate.py --model models/calibrated/me/model.pt

Requirements:
    pip install bleak anthropic

First run: macOS will prompt for Bluetooth permission. Allow it.
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
    ASL_CLASSES,
    BLE_CONTROL_UUID,
    BLE_EMG_CHAR_UUIDS,
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_MS,
    HOP_SAMPLES,
    LSTM_DROPOUT,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    MYO_ENABLE_EMG_CMD,
    N_CHANNELS,
    N_CLASSES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from src.models.lstm_classifier import LSTMClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = ROOT / "models" / "asl_emg_classifier.pt"
BUFFER_MAXLEN      = SAMPLE_RATE * 4   # 4 seconds rolling window
MYO_DEVICE_NAME    = "Myo"
LLM_PAUSE_S        = 1.8               # pause before sending letters to LLM
LLM_MIN_LETTERS    = 2                 # minimum letters before triggering LLM
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")

# Bandpass: 20-95 Hz (Myo @ 200 Hz, Nyquist = 100 Hz)
_NYQ = SAMPLE_RATE / 2.0
_SOS = butter(4, [20.0 / _NYQ, 95.0 / _NYQ], btype="band", output="sos")

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

R = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
VIOLET = "\033[95m"
DIM = "\033[2m"

def clr(text: str, code: str) -> str:
    return f"{code}{text}{R}"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(path: Path) -> LSTMClassifier:
    model = LSTMClassifier(
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
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
            print(f"  {clr('model', YELLOW)}  could not load weights ({exc})")
            print(f"           using random init — run calibrate_quick.py first")
    else:
        print(f"  {clr('model', YELLOW)}  no model file — random init")
        print(f"           run: python scripts/calibrate_quick.py")
    model.eval()
    return model


def predict(model: LSTMClassifier, window: np.ndarray) -> tuple[str, float]:
    x = torch.from_numpy(window).unsqueeze(0)   # (1, 40, 8)
    with torch.no_grad():
        probs = model.predict_proba(x)[0]        # (26,)
    idx = int(probs.argmax())
    return ASL_CLASSES[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# LLM translation
# ---------------------------------------------------------------------------

LLM_SYSTEM = (
    "You are an ASL-to-English interpreter. "
    "The user sends you a stream of fingerspelled letters (possibly with some errors). "
    "Reconstruct them into natural conversational English. "
    "Handle common letter substitution errors (e.g. 'HELLP' -> 'hello'). "
    "Output only the reconstructed English — no explanation, no quotes. "
    "Keep it concise and natural."
)


async def llm_translate(letters: list[str]) -> str:
    """Send a letter sequence to Claude and return reconstructed English."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=LLM_SYSTEM,
            messages=[{"role": "user", "content": "".join(letters)}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def render(letter: str, conf: float, letter_stream: list[str], sentence: str) -> None:
    bar_len = 20
    filled = int(conf * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    color = GREEN if conf >= 0.85 else (YELLOW if conf >= 0.65 else RED)
    stream_str = " ".join(letter_stream[-20:]) or DIM + "(waiting for input)" + R
    sentence_str = clr(sentence, BOLD + VIOLET) if sentence else DIM + "(will appear after you pause)" + R
    # Fixed 3-line display using cursor up
    print(
        f"\033[3A"                                                  # move up 3 lines
        f"\r  {clr(letter, BOLD)}  [{clr(bar, color)}] {clr(f'{conf:.0%}', color)}\033[K\n"
        f"\r  {clr('letters', DIM)}: {stream_str}\033[K\n"
        f"\r  {clr('echo', CYAN)}: {sentence_str}\033[K",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# BLE session
# ---------------------------------------------------------------------------

class MyoSession:
    def __init__(self, model: LSTMClassifier, use_llm: bool = True) -> None:
        self.model = model
        self.use_llm = use_llm
        self._buf: collections.deque[list[int]] = collections.deque(maxlen=BUFFER_MAXLEN)
        self._new_since_last = 0
        self._last_emit_ts = 0.0
        self._last_letter_ts = 0.0
        self._letter_stream: list[str] = []       # all confirmed letters this session
        self._pending_letters: list[str] = []     # letters since last LLM call
        self._sentence = ""                        # last LLM output
        self._llm_running = False

    def _on_emg(self, _handle: int, data: bytearray) -> None:
        if len(data) < 16:
            return
        for i in range(2):
            sample = list(struct.unpack_from("8b", data, i * 8))
            self._buf.append(sample)
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
        """Background task: flush pending letters to LLM after a pause."""
        while True:
            await asyncio.sleep(0.2)
            if not self.use_llm:
                continue
            if self._llm_running:
                continue
            if not self._pending_letters:
                continue
            if (time.monotonic() - self._last_letter_ts) < LLM_PAUSE_S:
                continue
            if len(self._pending_letters) < LLM_MIN_LETTERS:
                continue
            # Snapshot and clear pending buffer
            to_translate = list(self._pending_letters)
            self._pending_letters.clear()
            self._llm_running = True
            self._sentence = clr("...", DIM)
            translated = await llm_translate(to_translate)
            self._sentence = translated
            self._llm_running = False

    async def run(self) -> None:
        from bleak import BleakClient, BleakScanner

        print(f"  {clr('ble', CYAN)}  scanning for '{MYO_DEVICE_NAME}'...")
        device = await BleakScanner.find_device_by_name(MYO_DEVICE_NAME, timeout=10.0)
        if device is None:
            print(
                f"\n  {clr('error', RED)}  Myo not found.\n"
                "  Make sure it is:\n"
                "    1. Charged (LED pulses white)\n"
                "    2. Paired in System Settings > Bluetooth\n"
                "    3. Not connected to another app\n"
            )
            return

        print(f"  {clr('ble', CYAN)}  found {device.name} ({device.address})")

        async with BleakClient(device, timeout=15.0) as client:
            print(f"  {clr('ble', CYAN)}  connected")

            try:
                await client.write_gatt_char(BLE_CONTROL_UUID, MYO_ENABLE_EMG_CMD, response=True)
                print(f"  {clr('myo', CYAN)}  raw EMG enabled")
            except Exception as exc:
                print(f"  {clr('myo', YELLOW)}  enable cmd failed ({exc}), continuing")

            for uuid in BLE_EMG_CHAR_UUIDS:
                try:
                    await client.start_notify(uuid, self._on_emg)
                except Exception:
                    pass

            print()
            print(f"  {clr('live translation running', BOLD + GREEN)}")
            if self.use_llm:
                print(f"  {clr('LLM active', CYAN)} — pause signing to get natural English\n")
            else:
                print(f"  {clr('letters only mode', DIM)}\n")

            # Prime the 3-line display area
            print()
            print()
            print()

            hop_s = HOP_SAMPLES / SAMPLE_RATE
            llm_task = asyncio.create_task(self._llm_loop())

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
                transcript = " ".join(self._letter_stream)
                print(f"\n\n  {clr('session transcript', DIM)}: {transcript}")
                print(f"  {clr('last echo', VIOLET)}: {self._sentence}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    print()
    print(clr("  Echo  |  Live ASL Translation", BOLD))
    print(clr("  Myo BLE -> LSTM -> letters -> Claude -> English", DIM))
    print()
    model = load_model(Path(args.model))
    session = MyoSession(model, use_llm=not args.no_llm)
    try:
        await session.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to .pt model")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM translation layer")
    asyncio.run(main(parser.parse_args()))

#!/usr/bin/env python3
"""
calibrate_quick.py — 5-minute personal calibration using the Myo via BLE.

Walks you through signing each letter 3 times (3 sec each), then
fine-tunes the LSTM's final FC layer on your data. Saves a personal
model to models/calibrated/<user_id>/model.pt.

Usage (from project root):
    python scripts/calibrate_quick.py --user me

    # Use a subset of letters for a faster session:
    python scripts/calibrate_quick.py --user me --letters ABCDEFGHIJ

    # 5 reps per letter instead of 3 (more accurate):
    python scripts/calibrate_quick.py --user me --reps 5
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (
    ASL_CLASSES,
    BLE_CONTROL_UUID,
    BLE_EMG_CHAR_UUIDS,
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

BASE_MODEL_PATH = ROOT / "models" / "asl_emg_classifier.pt"
COLLECT_SECS    = 3       # seconds to collect per rep
FINE_TUNE_EPOCHS = 60
FINE_TUNE_LR    = 1e-3

_NYQ = SAMPLE_RATE / 2.0
_SOS = butter(4, [20.0 / _NYQ, 95.0 / _NYQ], btype="band", output="sos")

# Terminal
R = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RED = "\033[91m"

def clr(t: str, c: str) -> str:
    return f"{c}{t}{R}"


# ---------------------------------------------------------------------------
# BLE data collector
# ---------------------------------------------------------------------------

class MYOCollector:
    """BLE connection that collects raw EMG into a buffer on demand."""

    def __init__(self) -> None:
        self._buf: collections.deque[list[int]] = collections.deque(maxlen=SAMPLE_RATE * 10)
        self._collecting = False
        self._collected: list[list[int]] = []

    def _on_emg(self, _handle: int, data: bytearray) -> None:
        if len(data) < 16:
            return
        for i in range(2):
            sample = list(struct.unpack_from("8b", data, i * 8))
            self._buf.append(sample)
            if self._collecting:
                self._collected.append(sample)

    def start_collection(self) -> None:
        self._collected = []
        self._collecting = True

    def stop_collection(self) -> list[list[int]]:
        self._collecting = False
        data = list(self._collected)
        self._collected = []
        return data

    async def connect(self) -> "BleakClient":
        from bleak import BleakClient, BleakScanner
        print(f"  {clr('ble', CYAN)}  scanning for Myo...")
        device = await BleakScanner.find_device_by_name("Myo", timeout=10.0)
        if device is None:
            print(
                f"\n  {clr('error', RED)}  Myo not found.\n"
                "  Pair it in System Settings > Bluetooth first.\n"
            )
            sys.exit(1)
        print(f"  {clr('ble', CYAN)}  found {device.name} ({device.address})")
        client = BleakClient(device, timeout=15.0)
        await client.connect()
        print(f"  {clr('ble', CYAN)}  connected")
        try:
            await client.write_gatt_char(BLE_CONTROL_UUID, MYO_ENABLE_EMG_CMD, response=True)
        except Exception:
            pass
        for uuid in BLE_EMG_CHAR_UUIDS:
            try:
                await client.start_notify(uuid, self._on_emg)
            except Exception:
                pass
        return client


# ---------------------------------------------------------------------------
# Feature extraction for windows
# ---------------------------------------------------------------------------

def extract_windows(raw_samples: list[list[int]]) -> np.ndarray:
    """
    Convert raw int8 samples into a stack of (WINDOW_SAMPLES, N_CHANNELS) windows.
    Returns float32 array of shape (n_windows, WINDOW_SAMPLES, N_CHANNELS).
    """
    if len(raw_samples) < WINDOW_SAMPLES:
        return np.empty((0, WINDOW_SAMPLES, N_CHANNELS), dtype=np.float32)

    arr = np.array(raw_samples, dtype=np.float32) / 128.0   # (N, 8)
    arr = sosfilt(_SOS, arr, axis=0).astype(np.float32)

    windows = []
    for start in range(0, len(arr) - WINDOW_SAMPLES + 1, HOP_SAMPLES):
        windows.append(arr[start: start + WINDOW_SAMPLES])
    return np.array(windows, dtype=np.float32)   # (n, 40, 8)


# ---------------------------------------------------------------------------
# Fine-tuning: freeze all but final FC layer
# ---------------------------------------------------------------------------

def fine_tune(
    base_model: LSTMClassifier,
    X: np.ndarray,  # (N, 40, 8)
    y: np.ndarray,  # (N,)  int class indices
    epochs: int = FINE_TUNE_EPOCHS,
    lr: float = FINE_TUNE_LR,
) -> LSTMClassifier:
    """
    Freeze LSTM body, fine-tune only the final Linear layer.
    Returns a new model with updated weights.
    """
    model = copy.deepcopy(base_model)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the last linear layer in the FC head
    last_linear = None
    for module in model.fc.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        for param in last_linear.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()

    model.train()
    print(f"\n  {clr('fine-tuning', CYAN)}  {len(X)} windows, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_t).float().mean().item()
            print(
                f"    epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}  "
                f"acc={acc:.0%}",
                end="\r",
            )
    print()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main calibration flow
# ---------------------------------------------------------------------------

async def calibrate(args: argparse.Namespace) -> None:
    letters = list(args.letters.upper())
    reps = args.reps
    out_dir = ROOT / "models" / "calibrated" / args.user
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    base_model = LSTMClassifier(
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    if BASE_MODEL_PATH.exists():
        try:
            state = torch.load(BASE_MODEL_PATH, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            base_model.load_state_dict(state)
            print(f"  {clr('model', CYAN)}  loaded base weights")
        except Exception as exc:
            print(f"  {clr('model', YELLOW)}  random init ({exc})")
    base_model.eval()

    # BLE connection
    collector = MYOCollector()
    client = await collector.connect()

    print()
    print(clr(f"  Calibration for user: {args.user}", BOLD))
    print(f"  Letters: {' '.join(letters)}")
    print(f"  Reps per letter: {reps}  |  Duration per rep: {COLLECT_SECS}s")
    total_min = len(letters) * reps * (COLLECT_SECS + 2) / 60
    print(f"  Estimated time: ~{total_min:.1f} minutes\n")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    try:
        for letter in letters:
            class_idx = ASL_CLASSES.index(letter)
            letter_windows: list[np.ndarray] = []

            for rep in range(1, reps + 1):
                print(
                    f"  {clr(letter, BOLD + GREEN)}  rep {rep}/{reps}  "
                    f"— hold the sign in 3...",
                    end="\r",
                    flush=True,
                )
                await asyncio.sleep(1.0)
                print(
                    f"  {clr(letter, BOLD + GREEN)}  rep {rep}/{reps}  "
                    f"— hold the sign in 2...",
                    end="\r",
                    flush=True,
                )
                await asyncio.sleep(1.0)
                print(
                    f"  {clr(letter, BOLD + GREEN)}  rep {rep}/{reps}  "
                    f"— HOLD NOW!        ",
                    end="\r",
                    flush=True,
                )
                collector.start_collection()
                await asyncio.sleep(float(COLLECT_SECS))
                raw = collector.stop_collection()
                windows = extract_windows(raw)

                if len(windows) == 0:
                    print(f"\n  {clr('warning', YELLOW)}  no data for {letter} rep {rep} — skipping")
                else:
                    letter_windows.append(windows)
                    print(
                        f"  {clr(letter, BOLD + GREEN)}  rep {rep}/{reps}  "
                        f"— captured {len(windows)} windows   ",
                    )
                await asyncio.sleep(0.5)

            if letter_windows:
                X_letter = np.concatenate(letter_windows, axis=0)
                all_X.append(X_letter)
                all_y.append(np.full(len(X_letter), class_idx, dtype=np.int64))

    except KeyboardInterrupt:
        print(f"\n  {clr('interrupted', YELLOW)} — fine-tuning on collected data so far")
    finally:
        for uuid in BLE_EMG_CHAR_UUIDS:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass
        await client.disconnect()

    if not all_X:
        print(f"  {clr('error', RED)}  no data collected — exiting")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\n  collected {len(X)} total windows across {len(letters)} letters")

    # Fine-tune
    calibrated = fine_tune(base_model, X, y)

    # Save
    model_out = out_dir / "model.pt"
    torch.save(calibrated.state_dict(), model_out)
    print(f"\n  {clr('saved', GREEN)}  {model_out}")
    print(f"\n  Run live translation with your personal model:")
    print(f"  {clr(f'python scripts/live_translate.py --model {model_out}', CYAN)}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Quick Myo calibration for Echo ASL")
    parser.add_argument("--user", default="me", help="User ID for saving the model")
    parser.add_argument(
        "--letters",
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Letters to calibrate (default: A-Z)",
    )
    parser.add_argument("--reps", type=int, default=3, help="Reps per letter (default: 3)")
    args = parser.parse_args()

    print()
    print(clr("  Echo  |  Quick Calibration", BOLD))
    print(clr("  Myo BLE -> collect your EMG -> fine-tune LSTM", DIM))
    print()
    await calibrate(args)


if __name__ == "__main__":
    asyncio.run(main())

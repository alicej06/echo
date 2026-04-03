"""
WebSocket client for end-to-end testing of the EMG-ASL inference server.

Generates synthetic raw EMG frames (8-channel int16 little-endian binary), sends
them over the /stream WebSocket endpoint, and prints each prediction response.

Usage
-----
# Basic test — 20 windows, label "A" (label only affects synthetic noise bias):
python scripts/test_websocket.py

# Custom label and window count:
python scripts/test_websocket.py --url ws://192.168.1.x:8765/stream --label HELLO --windows 50

# Stress test — send 1000 windows as fast as possible:
python scripts/test_websocket.py --stress --windows 1000

Exit codes
----------
0  All windows sent and server remained connected throughout.
1  Connection failed or server closed unexpectedly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time

import numpy as np

try:
    import websockets
except ImportError:
    print("ERROR: 'websockets' not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants — must match src/utils/constants.py
# ---------------------------------------------------------------------------

DEFAULT_URL = "ws://localhost:8765/stream"
N_CHANNELS = 8
WINDOW_SIZE_SAMPLES = 40        # SAMPLE_RATE=200 Hz, WINDOW_SIZE_MS=200 ms
SEND_INTERVAL_S = 0.10          # 10 Hz — realistic BLE streaming rate
INT16_MAX = 32767
INT16_MIN = -32768


# ---------------------------------------------------------------------------
# Synthetic frame generator
# ---------------------------------------------------------------------------

def _make_raw_ble_frame(label: str, rng: np.random.Generator) -> bytes:
    """Return one binary BLE frame: 8-channel interleaved int16 LE.

    The server's EMGPipeline.ingest_bytes() expects exactly this layout:
        [ch0_s0, ch1_s0, ..., ch7_s0, ch0_s1, ..., ch7_s39]
    as little-endian signed 16-bit integers.

    We generate plausible EMG noise (zero-mean Gaussian, ~500 ADC units RMS)
    with a mild label-dependent DC offset so different labels produce slightly
    different synthetic signals.  This does not affect server behaviour — the
    random-weight ONNX baseline will predict randomly regardless — but it makes
    the frames look realistic in a packet capture.

    Parameters
    ----------
    label:
        ASL label string used to seed a repeatable per-label DC bias.
    rng:
        Shared NumPy random generator for reproducibility.
    """
    # Label-dependent bias: hash label to a small ADC offset per channel
    label_seed = sum(ord(c) for c in label)
    per_channel_bias = ((label_seed * np.arange(1, N_CHANNELS + 1)) % 200) - 100  # [-100, 100]

    # (WINDOW_SIZE_SAMPLES, N_CHANNELS) — Gaussian EMG noise
    samples = rng.normal(loc=0.0, scale=500.0, size=(WINDOW_SIZE_SAMPLES, N_CHANNELS))
    samples += per_channel_bias[np.newaxis, :]

    # Clip to int16 range and cast
    samples = np.clip(samples, INT16_MIN, INT16_MAX).astype(np.int16)

    # Pack as interleaved little-endian int16: row-major order is already
    # sample-major (each row = one time-point across all channels).
    return samples.tobytes()  # 40 * 8 * 2 = 640 bytes per frame


# ---------------------------------------------------------------------------
# Core send-receive loop
# ---------------------------------------------------------------------------

async def _run(url: str, label: str, n_windows: int, stress: bool) -> int:
    """Connect, stream frames, collect responses.  Returns exit code."""
    rng = np.random.default_rng(seed=42)

    print(f"Connecting to {url} ...")
    try:
        async with websockets.connect(url) as ws:  # type: ignore[attr-defined]
            print(f"Connected. Sending {n_windows} windows (label={label!r})"
                  f"{', STRESS MODE' if stress else f', interval={SEND_INTERVAL_S*1000:.0f}ms'}\n")

            latencies: list[float] = []
            responses_received = 0

            for i in range(1, n_windows + 1):
                frame = _make_raw_ble_frame(label, rng)
                t_send = time.perf_counter()
                await ws.send(frame)

                # The server only sends a response when the debounce interval
                # has elapsed AND confidence >= threshold.  With random weights
                # the baseline model rarely clears the threshold, so we use a
                # short receive timeout and treat silence as valid (server is
                # processing but suppressing low-confidence predictions).
                recv_label = "—"
                recv_conf = float("nan")
                latency_ms = float("nan")

                try:
                    raw_response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    t_recv = time.perf_counter()
                    latency_ms = (t_recv - t_send) * 1000.0
                    latencies.append(latency_ms)
                    responses_received += 1

                    data = json.loads(raw_response)
                    recv_label = data.get("label", "?")
                    recv_conf = data.get("confidence", float("nan"))

                    tag = "[server random-weight mode]" if i == 1 else ""
                    print(
                        f"Window {i:3d}/{n_windows}"
                        f" -> label={recv_label:<10s}"
                        f" conf={recv_conf:.2f}"
                        f" latency={latency_ms:.1f}ms"
                        f"  {tag}"
                    )
                except asyncio.TimeoutError:
                    # No response within 500 ms — server suppressed (debounce /
                    # low confidence).  This is normal with the baseline model.
                    print(
                        f"Window {i:3d}/{n_windows}"
                        f" -> (no response — debounce/confidence filter)"
                    )

                if not stress:
                    await asyncio.sleep(SEND_INTERVAL_S)

            # ------------------------------------------------------------------
            # Summary
            # ------------------------------------------------------------------
            print()
            if latencies:
                avg_lat = sum(latencies) / len(latencies)
                elapsed = n_windows * (SEND_INTERVAL_S if not stress else 0.0)
                # Use wall-clock throughput: approximate from total time if stress
                throughput = (
                    responses_received / elapsed
                    if elapsed > 0
                    else float("inf")
                )
                print(
                    f"Summary: {n_windows} windows sent, "
                    f"{responses_received} responses received, "
                    f"avg latency={avg_lat:.1f}ms"
                    + (f", throughput~{throughput:.1f} req/s" if elapsed > 0 else "")
                )
            else:
                print(
                    f"Summary: {n_windows} windows sent, "
                    f"0 responses received (all suppressed by server — "
                    f"normal with random-weight baseline model)."
                )

            return 0

    except (OSError, ConnectionRefusedError, websockets.exceptions.WebSocketException) as exc:
        print(f"\nERROR: Could not connect to {url!r}: {exc}", file=sys.stderr)
        print("Is the server running?  Start it with: ./start-server.sh", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end WebSocket test for the EMG-ASL inference server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"WebSocket URL to connect to (default: {DEFAULT_URL})",
    )
    p.add_argument(
        "--label",
        default="A",
        help="ASL label name for synthetic data bias (default: A)",
    )
    p.add_argument(
        "--windows",
        type=int,
        default=20,
        help="Number of windows to send (default: 20)",
    )
    p.add_argument(
        "--stress",
        action="store_true",
        help="Send windows as fast as possible with no inter-window sleep",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    exit_code = asyncio.run(
        _run(
            url=args.url,
            label=args.label,
            n_windows=args.windows,
            stress=args.stress,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

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
import concurrent.futures
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
from scripts.train_dtw import (
    PHRASES, TRAIN_REPS as DTW_TRAIN_REPS, MIN_REPS, NULL_CLASS, NULL_MIN_REPS,
    predict_dtw, train_dtw, save_dtw_model, load_dtw_model,
    save_phrase_recordings, load_phrase_recordings,
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

# --- DTW phrase segmentation ---
DTW_RMS_WINDOW        = 20    # samples to compute windowed RMS over (~100ms at 200Hz)
DTW_ONSET_RMS         = 18.0  # RMS threshold to START capturing a gesture
DTW_OFFSET_RMS        = 12.0  # RMS threshold to END capture (must drop below this)
DTW_MIN_QUIET         = 60    # consecutive quiet samples before gesture ends (300ms)
DTW_MIN_GESTURE       = 40    # minimum gesture length in samples (200ms)
DTW_MAX_GESTURE       = 1200  # maximum gesture buffer (6s)
DTW_CONFIDENCE_THRESH = 0.20  # minimum confidence to emit a phrase
TEACH_SAMPLES         = 400   # fixed-length recording for Teach Echo (2s at 200Hz)

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


# _ws_message_handler / _ws_connect_handler are set by _make_session
_ws_message_handler = None
_ws_connect_handler = None


async def _ws_server_handler(websocket) -> None:
    _ws_clients.add(websocket)
    print(f"  [ws]  client connected  ({len(_ws_clients)} total)")
    if _ws_connect_handler:
        try:
            await _ws_connect_handler(websocket)
        except Exception as exc:
            print(f"  [ws]  connect handler error: {exc}")
    try:
        async for raw in websocket:
            print(f"  [ws]  received: {raw[:120]}")
            if not _ws_message_handler:
                print(f"  [ws]  WARNING: no message handler set")
                continue
            try:
                msg = json.loads(raw)
                await _ws_message_handler(msg)
            except Exception as exc:
                print(f"  [ws]  handler error: {exc}")
    finally:
        _ws_clients.discard(websocket)
        print(f"  [ws]  client disconnected  ({len(_ws_clients)} remaining)")


async def _start_ws_server(port: int) -> None:
    try:
        import websockets  # type: ignore[import]
    except ImportError:
        print(f"  {clr('[ws]', YELLOW)}  websockets not installed — pip install websockets")
        return
    async with websockets.serve(_ws_server_handler, "0.0.0.0", port):
        print(f"  {clr('ws', CYAN)}  broadcast server → ws://localhost:{port}")
        await asyncio.Future()  # run forever

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

# Phrase-level LLM: receives ASL sign phrases (e.g. ["my", "name", "echo"])
# and reconstructs them into a natural English sentence.
LLM_PHRASE_SYSTEM = (
    "You are an ASL-to-English interpreter. "
    "You receive a sequence of ASL signs as individual words or short phrases. "
    "ASL omits words like 'is', 'are', 'the', 'a' — your job is to reconstruct "
    "natural spoken English. "
    "Examples: ['my', 'name', 'echo'] → 'My name is Echo.' "
    "['how are you'] → 'How are you?' "
    "['thank you', 'nice to meet you'] → 'Thank you, nice to meet you.' "
    "['hello', 'my', 'name', 'echo'] → 'Hello! My name is Echo.' "
    "['great'] → 'Great!' "
    "['what\\'s your name'] → 'What\\'s your name?' "
    "Output ONLY the English sentence. No quotes, no explanation. "
    "If the input is a single complete phrase, return it naturally capitalized."
)

LLM_PHRASE_PAUSE_S  = 2.5  # seconds of silence before constructing a sentence
LLM_PHRASE_MAX      = 8    # also flush if this many phrases are buffered


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


def _llm_call_with_system(system: str, text: str) -> str | None:
    """Call LLM backends in priority order with a custom system prompt."""
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                system=system,
                messages=[{"role": "user", "content": text}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            print(f"\n  {clr('[anthropic]', YELLOW)} {exc}")

    if OLLAMA_API_KEY:
        try:
            payload = json.dumps({
                "model": OLLAMA_CLOUD_MODEL, "stream": False,
                "messages": [{"role": "system", "content": system},
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
            pass

    try:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=OLLAMA_LOCAL_URL)
        resp = client.chat.completions.create(
            model=OLLAMA_LOCAL_MODEL, max_tokens=128,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": text}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    return None


_COMPLETE_PHRASES = {
    "how are you", "nice to meet you", "thank you",
    "what's your name", "great", "hello",
}
_QUESTION_PHRASES = {"how are you", "what's your name"}

def _phrase_ending(p: str) -> str:
    if p in _QUESTION_PHRASES:
        return "?"
    if p in ("hello", "great"):
        return "!"
    return "."

def _rule_based_translate(phrases: list[str]) -> str | None:
    """
    Fast, deterministic ASL→English reordering for the known vocabulary.
    Returns None if no rule matches (falls through to LLM).
    """
    seq = [p.lower().strip() for p in phrases]
    n = len(seq)

    # Single known complete phrase — just capitalise
    if n == 1:
        p = seq[0]
        if p in _COMPLETE_PHRASES:
            return p.capitalize() + _phrase_ending(p)
        if p == "echo":
            return "Echo."
        return p.capitalize() + "."

    # "my name echo" / "my name is echo"
    if "my" in seq and "name" in seq and "echo" in seq:
        greeting = "Hello! " if "hello" in seq else ""
        return f"{greeting}My name is Echo."

    # greeting + complete phrase combos
    parts = []
    if seq[0] == "hello":
        parts.append("Hello!")
        seq = seq[1:]
    for p in seq:
        if p in _COMPLETE_PHRASES:
            parts.append(p.capitalize() + _phrase_ending(p))
        elif p == "echo":
            parts.append("Echo.")
        elif p == "my":
            parts.append("my")
        elif p == "name":
            parts.append("name")
        else:
            parts.append(p.capitalize() + ".")
    if parts:
        return " ".join(parts)

    return None


async def llm_phrase_translate(phrases: list[str]) -> str:
    # Try fast rule-based first
    result = _rule_based_translate(phrases)
    if result:
        return result
    # Fall back to LLM for unknown combos
    text = ", ".join(phrases)
    return (
        _llm_call_with_system(LLM_PHRASE_SYSTEM, text)
        or " ".join(p.capitalize() for p in phrases)
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
    global _ws_message_handler, _ws_connect_handler

    # DyFAV letter model — kept as None (words-only mode; WS train_model handler still references it)
    dyfav_model = [None]

    # Load DTW phrase model
    dtw_model = [load_dtw_model(user_id)]
    if dtw_model[0]:
        phrases_loaded = dtw_model[0].get("phrases", [])
        print(f"  {clr('dtw', CYAN)}  loaded phrase model ({len(phrases_loaded)} phrases) for '{user_id}'")
    else:
        print(f"  {clr('dtw', YELLOW)}  no phrase model for '{user_id}' — run --train-words to build one")

    # DTW phrase training recordings — load persisted recordings from disk
    phrase_recordings: dict[str, list] = load_phrase_recordings(user_id)
    if phrase_recordings:
        total_recs = sum(len(v) for v in phrase_recordings.values())
        print(f"  {clr('dtw', CYAN)}  loaded {total_recs} saved phrase recordings for '{user_id}'")

    # DTW gesture segmentation state
    seg_buf: list       = []
    seg_quiet_count     = [0]
    seg_active          = [False]
    display_tick        = [0]   # rate-limit terminal status updates
    ws_train_phrase     = [None]   # set while waiting for a training gesture via WS
    ws_thinking         = [False]  # True while DTW inference is running
    # Teach Echo state — fixed 2s recording for custom gestures
    teach_collecting    = [False]
    teach_word_ref      = [None]   # the word being taught
    teach_buf: list     = []

    # Shared thread pool for DTW inference (avoid creating one per gesture)
    _dtw_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _rms_emg(rows: list) -> float:
        if len(rows) < 2:
            return 0.0
        arr = np.array(rows[-DTW_RMS_WINDOW:], dtype=np.float32)[:, :8]
        return float(np.sqrt(np.mean(arr ** 2)))

    # Training state
    # recordings[letter] = list of (n_samples, 17) arrays
    recordings: dict[str, list[np.ndarray]] = {}
    collecting         = [False]
    collect_letter     = [None]
    collect_buf: list  = []   # rows of 17 values being collected

    # Inference state
    # sync_buf: rolling deque of 17-column rows (EMG8 + IMU9), carry-forward per sample.
    # This matches exactly how collect_buf is built during training.
    sync_buf  = collections.deque(maxlen=COLLECT_SAMPLES * 4)
    imu_state = [np.zeros(9, dtype=np.float32)]  # last known IMU (carry-forward)
    new_emg   = [0]

    letter_stream:  list[str] = []
    pending_letters: list[str] = []
    sentence        = [""]
    last_emit_ts    = [0.0]
    last_letter_ts  = [0.0]
    llm_running     = [False]
    mode            = ["recognition"]  # "recognition" | "training"

    # Phrase-level LLM accumulation
    pending_phrases:  list[str] = []
    last_phrase_ts    = [0.0]
    phrase_llm_running = [False]

    # Gesture stability state
    stable_letter   = [None]   # letter currently being held
    stable_count    = [0]      # how many consecutive frames it's been top
    last_emitted    = [None]   # last letter we actually emitted (require change to re-emit)

    def _build_sync_array() -> np.ndarray:
        """Return last COLLECT_SAMPLES rows from sync_buf.
        Each row already has (EMG8 + IMU9) captured at the right time, matching training."""
        return np.array(list(sync_buf)[-COLLECT_SAMPLES:], dtype=np.float32)

    def _infer() -> None:
        if dyfav_model[0] is None:
            return
        if len(sync_buf) < COLLECT_SAMPLES:
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

            collecting[0]     = True
            collect_letter[0] = letter
            collect_buf.clear()
            mode[0] = f"collecting {letter.upper()}"
            print(f"\n  {clr('train', CYAN)}  collecting '{letter.upper()}'...")

            # Watchdog: if EMG stops flowing (Myo drops), un-stick the UI after 5s
            async def _collection_watchdog(ltr: str) -> None:
                await asyncio.sleep(5.0)
                if collecting[0] and collect_letter[0] == ltr:
                    collecting[0] = False
                    mode[0] = "recognition"
                    await _ws_broadcast({"type": "error", "message": f"Recording timed out — is the Myo streaming?"})
            asyncio.create_task(_collection_watchdog(letter))

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
            recs_snapshot = {k: list(v) for k, v in recordings.items()}

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                model = await loop.run_in_executor(
                    pool, train_from_recordings, recs_snapshot
                )

            import joblib
            model["user_id"] = user_id
            dyfav_model[0] = model
            user_dir = MODELS_DIR / f"user_{user_id}"
            user_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, user_dir / "dyfav_model.pkl")
            recordings.clear()
            mode[0] = "recognition"
            print(f"  {clr('train', GREEN)}  model saved → models/user_{user_id}/dyfav_model.pkl")
            await _ws_broadcast({"type": "model_ready", "user_id": user_id})

        elif mtype == "train_phrase":
            phrase = str(msg.get("phrase", "")).lower()
            if phrase not in PHRASES:
                await _ws_broadcast({"type": "error", "message": f"Unknown phrase: {phrase}"})
                return
            if ws_train_phrase[0] or collecting[0]:
                await _ws_broadcast({"type": "error", "message": "Already recording"})
                return
            ws_train_phrase[0] = phrase
            mode[0] = f"waiting: {phrase}"
            print(f"\n  {clr('train', CYAN)}  ready — sign '{phrase}'  (gesture segmentation active)")

            async def _phrase_watchdog(p: str) -> None:
                await asyncio.sleep(12.0)
                if ws_train_phrase[0] == p:
                    ws_train_phrase[0] = None
                    mode[0] = "recognition"
                    await _ws_broadcast({"type": "error", "message": f"Phrase recording timed out — no gesture detected"})
            asyncio.create_task(_phrase_watchdog(phrase))

        elif mtype == "train_null":
            if ws_train_phrase[0] or collecting[0]:
                await _ws_broadcast({"type": "error", "message": "Already recording"})
                return
            ws_train_phrase[0] = NULL_CLASS
            mode[0] = f"waiting: null"
            print(f"\n  {clr('train', CYAN)}  ready — do a random arm movement (not a sign)")

            async def _null_watchdog() -> None:
                await asyncio.sleep(12.0)
                if ws_train_phrase[0] == NULL_CLASS:
                    ws_train_phrase[0] = None
                    mode[0] = "recognition"
                    await _ws_broadcast({"type": "error", "message": "Null recording timed out"})
            asyncio.create_task(_null_watchdog())

        elif mtype == "train_phrases_model":
            # Merge in-memory recordings with any previously persisted ones
            disk_recs = load_phrase_recordings(user_id)
            merged: dict[str, list] = {**disk_recs}
            for p, recs in phrase_recordings.items():
                merged.setdefault(p, []).extend(recs)

            missing = [p for p in PHRASES if len(merged.get(p, [])) < 3]
            if missing:
                await _ws_broadcast({
                    "type": "error",
                    "message": f"Need at least 3 recs each — missing: {missing[:3]}",
                })
                return
            print(f"\n  {clr('train', CYAN)}  training DTW phrase model for '{user_id}'...")
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(_dtw_executor, train_dtw, merged)
            dtw_model[0] = model
            save_dtw_model(model, user_id)
            save_phrase_recordings(merged, user_id)
            phrase_recordings.clear()
            mode[0] = "recognition"
            cv_acc = model.get("cv_acc")
            acc_str = f"  CV acc: {cv_acc:.1%}" if cv_acc is not None else ""
            print(f"  {clr('train', GREEN)}  phrase model saved for '{user_id}'{acc_str}")
            await _ws_broadcast({
                "type": "dtw_model_ready",
                "user_id": user_id,
                "cv_accuracy": round(cv_acc, 4) if cv_acc is not None else None,
            })

        elif mtype == "teach_record":
            word = str(msg.get("word", "")).strip().lower()
            if not word:
                await _ws_broadcast({"type": "error", "message": "Word cannot be empty"})
                return
            if teach_collecting[0] or collecting[0] or ws_train_phrase[0]:
                await _ws_broadcast({"type": "error", "message": "Already recording"})
                return
            teach_collecting[0] = True
            teach_word_ref[0]   = word
            teach_buf.clear()
            mode[0] = f"teach: {word}"
            print(f"\n  {clr('teach', VIOLET)}  recording '{word}'  ({TEACH_SAMPLES} samples)...")

            async def _teach_watchdog(w: str) -> None:
                await asyncio.sleep(6.0)
                if teach_collecting[0] and teach_word_ref[0] == w:
                    teach_collecting[0] = False
                    teach_word_ref[0]   = None
                    teach_buf.clear()
                    mode[0] = "recognition"
                    await _ws_broadcast({"type": "error", "message": "Teach recording timed out"})
            asyncio.create_task(_teach_watchdog(word))

        elif mtype == "teach_train":
            word = str(msg.get("word", "")).strip().lower()
            if not word:
                await _ws_broadcast({"type": "error", "message": "No word specified"})
                return
            # Disk is already up to date (each rep merges before saving).
            # Using disk as the single source of truth avoids double-counting
            # in-memory recs that were already flushed to disk.
            disk_recs = load_phrase_recordings(user_id)
            merged: dict[str, list] = dict(disk_recs)
            print(f"  {clr('teach', VIOLET)}  training with {sum(len(v) for v in merged.values())} "
                  f"total reps across {len(merged)} gesture(s)")
            if len(merged.get(word, [])) < MIN_REPS:
                await _ws_broadcast({
                    "type": "error",
                    "message": f"Need {MIN_REPS} reps of '{word}' — have {len(merged.get(word, []))}",
                })
                return
            trainable = {
                p: v for p, v in merged.items()
                if len(v) >= (NULL_MIN_REPS if p == NULL_CLASS else MIN_REPS)
            }
            if len(trainable) < 2:
                await _ws_broadcast({
                    "type": "error",
                    "message": "Need at least 2 gesture types trained before adding a new one",
                })
                return
            print(f"\n  {clr('teach', VIOLET)}  training model with new gesture '{word}'...")
            model = await asyncio.get_running_loop().run_in_executor(_dtw_executor, train_dtw, trainable)
            dtw_model[0] = model
            save_dtw_model(model, user_id)
            save_phrase_recordings(merged, user_id)
            phrase_recordings.clear()
            mode[0] = "recognition"
            cv_acc = model.get("cv_acc")
            print(f"  {clr('teach', GREEN)}  model updated — '{word}' added"
                  + (f"  cv_acc={cv_acc:.1%}" if cv_acc else ""))
            await _ws_broadcast({
                "type": "teach_model_ready",
                "word": word,
                "cv_accuracy": round(cv_acc, 4) if cv_acc is not None else None,
            })

        elif mtype == "correction":
            letter = str(msg.get("letter", "")).lower()
            print(f"\n  {clr('correction', YELLOW)}  user says: {letter.upper()}")

    async def _run_dtw(raw_gesture: np.ndarray) -> None:
        """Run DTW phrase inference in thread pool, broadcast result."""
        ws_thinking[0] = True
        if ws_port:
            asyncio.create_task(_ws_broadcast({"type": "gesture_state", "state": "thinking"}))
        loop = asyncio.get_event_loop()
        phrase, confidence, scores = await loop.run_in_executor(
            _dtw_executor,
            lambda: predict_dtw(dtw_model[0], raw_gesture, return_scores=True),
        )
        ws_thinking[0] = False
        if phrase is None:
            # Null class — suppress output silently
            print(f"\r  {clr('--', DIM)}  [null / background]  ({confidence:.0%})          \n",
                  flush=True)
            if ws_port:
                await _ws_broadcast({"type": "gesture_state", "state": "idle", "rms": 0.0})
            return
        if confidence < DTW_CONFIDENCE_THRESH:
            print(f"\r  {clr('?', DIM)}  {phrase}  ({confidence:.0%}) — below threshold          \n",
                  flush=True)
            if ws_port:
                await _ws_broadcast({"type": "gesture_state", "state": "idle", "rms": 0.0})
            return
        print(f"\r  {clr(phrase.upper(), BOLD + GREEN)}  ({confidence:.0%})          \n", flush=True)
        if ws_port:
            await _ws_broadcast({
                "type":       "phrase",
                "phrase":     phrase,
                "confidence": round(confidence, 4),
                "scores":     {p: round(d, 3) for p, d in scores.items()},
            })
        # Feed into phrase-level LLM accumulator
        pending_phrases.append(phrase)
        last_phrase_ts[0] = time.monotonic()

    _ws_message_handler = _handle_ws_message

    async def _handle_ws_connect(websocket) -> None:
        """Send current phrase training state and model status to a newly connected client."""
        counts = {p: len(phrase_recordings.get(p, [])) for p in PHRASES}
        await websocket.send(json.dumps({
            "type": "phrase_train_status",
            "counts": counts,
        }))
        if dtw_model[0]:
            cv_acc = dtw_model[0].get("cv_acc")
            await websocket.send(json.dumps({
                "type": "dtw_model_ready",
                "user_id": user_id,
                "cv_accuracy": round(cv_acc, 4) if cv_acc is not None else None,
            }))

    _ws_connect_handler = _handle_ws_connect

    from myo import MyoClient
    from myo.types import (
        EMGData, IMUData, EMGMode, IMUMode, ClassifierMode,
        AggregatedData, ClassifierEvent, EMGDataSingle, FVData, MotionEvent,
    )

    class EchoMyo(MyoClient):

        async def on_emg_data(self, emg: EMGData) -> None:
            for sample in (emg.sample1, emg.sample2):
                row = np.array(list(sample), dtype=np.float32)
                # Combine EMG with carry-forward IMU and store in rolling buffer.
                # sync_buf is used for both inference and training collection so
                # the data format is identical in both cases.
                full_row = np.concatenate([row, imu_state[0]])
                sync_buf.append(full_row)

                # Teach Echo: fixed-length timed recording
                if teach_collecting[0]:
                    teach_buf.append(full_row)
                    # Stream raw EMG for the live visualiser
                    if ws_port and len(teach_buf) % 4 == 0:
                        emg_vals = [int(v) for v in row[:8]]
                        asyncio.create_task(_ws_broadcast({
                            "type": "teach_emg",
                            "emg": emg_vals,
                        }))
                    if len(teach_buf) >= TEACH_SAMPLES:
                        raw_t  = np.array(teach_buf[:TEACH_SAMPLES], dtype=np.float32)
                        word   = teach_word_ref[0]
                        teach_collecting[0] = False
                        teach_word_ref[0]   = None
                        teach_buf.clear()
                        mode[0] = "recognition"
                        phrase_recordings.setdefault(word, []).append(raw_t)
                        # BUG FIX: merge with existing disk recordings before saving so
                        # previously taught words are not overwritten.
                        _disk = load_phrase_recordings(user_id)
                        _disk[word] = phrase_recordings[word]   # update only this word
                        save_phrase_recordings(_disk, user_id)
                        count = len(phrase_recordings[word])
                        total_words = len(_disk)
                        print(f"  {clr('teach', VIOLET)}  '{word}' rep {count} saved  "
                              f"(library has {total_words} gesture(s) on disk)")
                        if ws_port:
                            asyncio.create_task(_ws_broadcast({
                                "type": "teach_ack",
                                "word": word,
                                "count": count,
                            }))

                if collecting[0]:
                    collect_buf.append(full_row)

                    if len(collect_buf) >= COLLECT_SAMPLES:
                        raw    = np.array(collect_buf[:COLLECT_SAMPLES], dtype=np.float32)
                        key    = collect_letter[0]
                        collecting[0] = False
                        mode[0] = "recognition"

                        # Letter training recording (phrase training uses gesture segmentation)
                        recordings.setdefault(key, []).append(raw)
                        count = len(recordings[key])
                        print(f"  {clr('train', CYAN)}  '{key.upper()}' recorded ({count}/{TRAIN_REPS_NEEDED})")
                        if ws_port:
                            asyncio.create_task(_ws_broadcast({
                                "type": "train_ack",
                                "letter": key,
                                "count": count,
                                "needed": TRAIN_REPS_NEEDED,
                            }))

                # --- DTW gesture segmentation (runs continuously) ---
                if not collecting[0]:
                    rms = _rms_emg(seg_buf if seg_active[0] else list(sync_buf))

                    if not seg_active[0]:
                        if rms >= DTW_ONSET_RMS:
                            seg_active[0]      = True
                            seg_quiet_count[0] = 0
                            seg_buf.clear()
                            seg_buf.append(full_row)
                            # New line so gesture progress doesn't overwrite idle bar
                            print(f"\n  {clr('>>', BOLD + GREEN)}  gesture — signing...", flush=True)
                    else:
                        seg_buf.append(full_row)
                        if rms < DTW_OFFSET_RMS:
                            seg_quiet_count[0] += 1
                        else:
                            seg_quiet_count[0] = 0

                        gesture_done = (
                            seg_quiet_count[0] >= DTW_MIN_QUIET
                            or len(seg_buf) >= DTW_MAX_GESTURE
                        )
                        if gesture_done:
                            seg_active[0] = False
                            if len(seg_buf) >= DTW_MIN_GESTURE:
                                raw_g = np.array(seg_buf, dtype=np.float32)
                                seg_buf.clear()
                                if ws_train_phrase[0]:
                                    # Save as phrase training recording
                                    phrase = ws_train_phrase[0]
                                    ws_train_phrase[0] = None
                                    mode[0] = "recognition"
                                    phrase_recordings.setdefault(phrase, []).append(raw_g)
                                    save_phrase_recordings(phrase_recordings, user_id)
                                    count = len(phrase_recordings[phrase])
                                    print(f"\r  {clr('train', CYAN)}  '{phrase}' recorded ({count}/{DTW_TRAIN_REPS})          \n",
                                          flush=True)
                                    if ws_port:
                                        asyncio.create_task(_ws_broadcast({
                                            "type": "train_phrase_ack",
                                            "phrase": phrase,
                                            "count": count,
                                            "needed": DTW_TRAIN_REPS,
                                        }))
                                elif dtw_model[0] is not None:
                                    print(f"\r  {clr('..', CYAN)}  thinking...  ({len(raw_g)} frames)          ",
                                          end="", flush=True)
                                    asyncio.create_task(_run_dtw(raw_g))
                                # else: no model and not training — ignore gesture
                            else:
                                # Too short — discard silently, return to idle bar
                                seg_buf.clear()
                                if ws_train_phrase[0]:
                                    print(f"\r  {clr('train', YELLOW)}  gesture too short — sign again          ",
                                          end="", flush=True)
                                else:
                                    print(f"\r  {clr('listening', DIM)}  (gesture too short, ignored)          ",
                                          end="", flush=True)

                # Rate-limited idle/capture status bar (~10 Hz)
                display_tick[0] += 2
                if display_tick[0] >= 20:
                    display_tick[0] = 0
                    rms_now = _rms_emg(list(sync_buf))
                    if seg_active[0]:
                        frames   = len(seg_buf)
                        bar_fill = min(frames // 12, 20)
                        bar      = clr("=" * bar_fill, GREEN) + clr("-" * (20 - bar_fill), DIM)
                        print(f"\r  {clr('>>', BOLD + GREEN)}  [{bar}]  {frames} fr  RMS:{rms_now:5.1f}  ",
                              end="", flush=True)
                        if ws_port:
                            asyncio.create_task(_ws_broadcast({
                                "type": "gesture_state", "state": "capturing",
                                "frames": frames, "rms": round(rms_now, 1),
                            }))
                    elif ws_thinking[0]:
                        pass  # thinking broadcast already sent; don't spam
                    elif ws_train_phrase[0]:
                        thresh_fill = min(int(rms_now / DTW_ONSET_RMS * 12), 12)
                        bar         = clr("|" * thresh_fill, CYAN) + clr("." * (12 - thresh_fill), DIM)
                        print(f"\r  {clr('rec>', BOLD + CYAN)}  [{bar}]  sign: {ws_train_phrase[0]}  ",
                              end="", flush=True)
                        if ws_port:
                            asyncio.create_task(_ws_broadcast({
                                "type": "gesture_state", "state": "idle",
                                "rms": round(rms_now, 1),
                            }))
                    elif dtw_model[0] is not None:
                        thresh_fill = min(int(rms_now / DTW_ONSET_RMS * 12), 12)
                        bar         = clr("|" * thresh_fill, YELLOW) + clr("." * (12 - thresh_fill), DIM)
                        print(f"\r  {clr('listening', DIM)}  [{bar}]  RMS:{rms_now:5.1f}  ",
                              end="", flush=True)
                        if ws_port:
                            asyncio.create_task(_ws_broadcast({
                                "type": "gesture_state", "state": "idle",
                                "rms": round(rms_now, 1),
                            }))

            new_emg[0] += 2

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

    async def _phrase_llm_loop() -> None:
        """
        Accumulate recognised phrase tokens and flush them to the LLM when
        the user pauses (LLM_PHRASE_PAUSE_S silence) or the buffer is full.

        E.g. ["my", "name", "echo"] → "My name is Echo."
        """
        while True:
            await asyncio.sleep(0.2)
            if not use_llm or phrase_llm_running[0] or not pending_phrases:
                continue
            silence = time.monotonic() - last_phrase_ts[0]
            if silence < LLM_PHRASE_PAUSE_S and len(pending_phrases) < LLM_PHRASE_MAX:
                continue
            to_translate = list(pending_phrases)
            pending_phrases.clear()
            phrase_llm_running[0] = True
            print(f"\r  {clr('llm', VIOLET)}  constructing sentence from: {to_translate}          ",
                  flush=True)
            if ws_port:
                asyncio.create_task(_ws_broadcast({"type": "sentence_building", "phrases": to_translate}))
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: _rule_based_translate(to_translate)
                    or _llm_call_with_system(LLM_PHRASE_SYSTEM, ", ".join(to_translate))
                    or " ".join(p.capitalize() for p in to_translate)
            )
            sentence[0] = result
            phrase_llm_running[0] = False
            print(f"\r  {clr('echo', BOLD + VIOLET)}  {result}          \n", flush=True)
            if ws_port:
                await _ws_broadcast({"type": "sentence", "text": result})

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
        dtw_status = clr('loaded', GREEN) if dtw_model[0] else clr('NONE — run --train-words first', YELLOW)
        print(f"  {clr('Echo running', BOLD + GREEN)}"
              f"  |  user: {clr(user_id, CYAN)}"
              f"  |  phrase model: {dtw_status}")
        print()

        try:
            asyncio.create_task(_phrase_llm_loop())
            await client.start()
            while True:
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await client.stop()
            await client.disconnect()
            if ws_port:
                await _ws_broadcast({"type": "status", "connected": False, "device": device_mac or "Myo"})

    return run

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _inspect(device_mac: str) -> None:
    """
    Stream raw Myo sensor values to the terminal so you can verify
    channel ordering, scale, and orientation before training.

    Prints one line per EMG packet:
      EMG[0..7]  |  quat(w x y)  gyro(x y z)  accel(x y z)
    """
    from myo import MyoClient
    from myo.types import EMGData, IMUData, EMGMode, IMUMode, ClassifierMode

    imu_state = [np.zeros(9, dtype=np.float32)]
    n = [0]

    class InspectMyo(MyoClient):
        async def on_emg_data(self, emg: EMGData) -> None:
            n[0] += 1
            if n[0] % 20 != 0:   # print ~10 lines/sec (200Hz / 20)
                return
            e = list(emg.sample1)
            imu = imu_state[0]
            emg_str   = "  ".join(f"{v:4d}" for v in e)
            quat_str  = "  ".join(f"{v:7.4f}" for v in imu[0:3])
            gyro_str  = "  ".join(f"{v:8.3f}" for v in imu[3:6])
            accel_str = "  ".join(f"{v:8.3f}" for v in imu[6:9])
            print(
                f"EMG [{emg_str}]"
                f"  |  quat [{quat_str}]"
                f"  |  gyro [{gyro_str}]"
                f"  |  accel [{accel_str}]"
            )

        async def on_imu_data(self, imu: IMUData) -> None:
            try:
                o = imu.orientation
                g = imu.gyroscope
                a = imu.accelerometer
                imu_state[0] = np.array(
                    [o.w, o.x, o.y, g[0], g[1], g[2], a[0], a[1], a[2]],
                    dtype=np.float32,
                )
            except Exception:
                pass

        async def on_emg_data_aggregated(self, eds) -> None: pass
        async def on_aggregated_data(self, ad) -> None: pass
        async def on_fv_data(self, fvd) -> None: pass
        async def on_motion_event(self, me) -> None: pass
        async def on_classifier_event(self, ce) -> None: pass

    print(clr("\n  Inspect mode — streaming raw Myo values (Ctrl+C to stop)\n", CYAN))
    print("  EMG [0  1  2  3  4  5  6  7]"
          "  |  quat [w       x       y      ]"
          "  |  gyro [x         y         z       ]"
          "  |  accel [x         y         z      ]")
    print("  " + "-" * 110)

    client = await InspectMyo.with_device(mac=device_mac or None)
    await client.setup(
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_ALL,
        classifier_mode=ClassifierMode.DISABLED,
    )
    try:
        await client.start()
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await client.stop()
        await client.disconnect()


async def _train_terminal(
    user_id: str,
    device_mac: str,
    reps: int = TRAIN_REPS_NEEDED,
) -> None:
    """
    Terminal-only training mode.  No frontend or WebSocket needed.

    For each of the 26 letters:
      1. Print the ASL hint
      2. Wait for Enter (user forms the gesture)
      3. Record COLLECT_SAMPLES EMG+IMU samples
      4. Repeat `reps` times
    Then train and save the model.
    """
    from myo import MyoClient
    from myo.types import EMGData, IMUData, EMGMode, IMUMode, ClassifierMode

    print(clr(f"\n  Training mode — user '{user_id}'  ({reps} reps per letter)", BOLD))
    print(clr("  Connect, then hold each pose when prompted and press Enter.\n", DIM))

    emg_buf:  list = []
    imu_state = [np.zeros(9, dtype=np.float32)]
    collecting = [False]
    collect_done = [False]

    class TrainMyo(MyoClient):
        async def on_emg_data(self, emg: EMGData) -> None:
            if not collecting[0]:
                return
            for sample in (emg.sample1, emg.sample2):
                row = np.array(list(sample), dtype=np.float32)
                full_row = np.concatenate([row, imu_state[0]])
                emg_buf.append(full_row)
                if len(emg_buf) >= COLLECT_SAMPLES:
                    collecting[0] = False
                    collect_done[0] = True

        async def on_imu_data(self, imu: IMUData) -> None:
            try:
                o = imu.orientation
                g = imu.gyroscope
                a = imu.accelerometer
                imu_state[0] = np.array(
                    [o.w, o.x, o.y, g[0], g[1], g[2], a[0], a[1], a[2]],
                    dtype=np.float32,
                )
            except Exception:
                pass

        async def on_emg_data_aggregated(self, eds) -> None: pass
        async def on_aggregated_data(self, ad) -> None: pass
        async def on_fv_data(self, fvd) -> None: pass
        async def on_motion_event(self, me) -> None: pass
        async def on_classifier_event(self, ce) -> None: pass

    print(clr("  Connecting to Myo...", CYAN))
    client = await TrainMyo.with_device(mac=device_mac or None)
    await client.setup(
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_ALL,
        classifier_mode=ClassifierMode.DISABLED,
    )
    print(clr("  Connected. Starting training.\n", GREEN))

    HINTS = {
        'a': "Fist, thumb to side",   'b': "Flat hand, thumb in",
        'c': "Curved fingers+thumb",  'd': "Index up, others to thumb",
        'e': "Fingers bent to palm",  'f': "Thumb+index touch, others spread",
        'g': "Index+thumb sideways",  'h': "Index+middle sideways",
        'i': "Pinky up from fist",    'j': "Pinky up, trace J",
        'k': "Index+middle+thumb K",  'l': "Thumb+index right angle",
        'm': "Three fingers over thumb", 'n': "Two fingers over thumb",
        'o': "Fingers+thumb form O",  'p': "K-shape pointing down",
        'q': "G-shape pointing down", 'r': "Index+middle crossed",
        's': "Fist, thumb over fingers", 't': "Thumb between index+middle",
        'u': "Index+middle together", 'v': "Index+middle spread (peace)",
        'w': "Three fingers spread",  'x': "Index hooked",
        'y': "Thumb+pinky spread",    'z': "Index traces Z",
    }

    recordings: dict[str, list[np.ndarray]] = {}
    loop = asyncio.get_event_loop()

    async def _ainput(prompt: str) -> str:
        """Non-blocking input — runs in a thread so the event loop stays alive
        and Myo BLE callbacks keep firing while we wait for Enter."""
        return await loop.run_in_executor(None, input, prompt)

    async def _record_one() -> np.ndarray:
        """Countdown then capture COLLECT_SAMPLES rows from the live buffer."""
        # 3-2-1 countdown: event loop stays alive, Myo keeps streaming
        for countdown in (3, 2, 1):
            print(f"\r    {clr(str(countdown), YELLOW)}...", end="", flush=True)
            await asyncio.sleep(0.4)
        print(f"\r    {clr('* REC', RED)}    ", end="", flush=True)

        emg_buf.clear()
        collecting[0]  = True
        collect_done[0] = False
        while not collect_done[0]:
            await asyncio.sleep(0.005)
        print(f"\r    {clr('done', GREEN)}    ")
        return np.array(emg_buf[:COLLECT_SAMPLES], dtype=np.float32)

    client_task = asyncio.create_task(client.start())
    # Let the Myo settle for a moment before training starts
    await asyncio.sleep(1.0)

    for letter in ALL_LETTERS:
        hint = HINTS.get(letter, "")
        print(f"\n  {clr(letter.upper(), BOLD + CYAN)}  {clr(hint, DIM)}")
        for rep in range(1, reps + 1):
            await _ainput(f"    Rep {rep}/{reps} — hold the pose, press Enter ")
            raw = await _record_one()
            recordings.setdefault(letter, []).append(raw)
        print()

    client_task.cancel()
    try:
        await client_task
    except (asyncio.CancelledError, Exception):
        pass
    await client.stop()
    await client.disconnect()

    print(clr("\n  Training DyFAV model...", CYAN))
    import joblib, concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        model = await loop.run_in_executor(pool, train_from_recordings, recordings)
    model["user_id"] = user_id
    user_dir = MODELS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, user_dir / "dyfav_model.pkl")
    print(clr(f"  Model saved → models/user_{user_id}/dyfav_model.pkl", GREEN))
    print(clr("  Run without --train to start recognition.\n", DIM))


async def _train_null_terminal(
    user_id: str,
    device_mac: str,
    reps: int = 30,
) -> None:
    """Record null/background gestures from the terminal."""
    from myo import MyoClient
    from myo.types import EMGData, IMUData, EMGMode, IMUMode, ClassifierMode

    print(clr(f"\n  Null recording — user '{user_id}'  ({reps} reps)", BOLD))
    print(clr("  Each rep: press Enter, then do any non-sign movement for ~2s.\n"
              "  Vary every rep: arm resting, reaching, waving, talking gestures.\n", DIM))

    imu_state = [np.zeros(9, dtype=np.float32)]
    buf: list = []
    collecting = [False]

    class NullMyo(MyoClient):
        async def on_emg_data(self, emg: EMGData) -> None:
            for sample in (emg.sample1, emg.sample2):
                row = np.array(list(sample), dtype=np.float32)
                full_row = np.concatenate([row, imu_state[0]])
                if collecting[0]:
                    buf.append(full_row)

        async def on_imu_data(self, imu: IMUData) -> None:
            try:
                o = imu.orientation
                g = imu.gyroscope
                a = imu.accelerometer
                imu_state[0] = np.array([
                    o.w, o.x, o.y,
                    g[0], g[1], g[2],
                    a[0], a[1], a[2],
                ], dtype=np.float32)
            except Exception:
                pass

        async def on_aggregated_data(self, ad) -> None: pass
        async def on_classifier_event(self, ce) -> None: pass
        async def on_emg_data_aggregated(self, eds) -> None: pass
        async def on_fv_data(self, fvd) -> None: pass
        async def on_motion_event(self, me) -> None: pass

    client = await NullMyo.with_device(mac=device_mac or None)
    await client.setup(emg_mode=EMGMode.SEND_EMG, imu_mode=IMUMode.SEND_ALL,
                       classifier_mode=ClassifierMode.DISABLED)

    recordings = load_phrase_recordings(user_id)
    loop = asyncio.get_running_loop()
    client_task = asyncio.create_task(client.start())
    await asyncio.sleep(1.0)

    for i in range(reps):
        await loop.run_in_executor(None, input, f"  Rep {i+1}/{reps} — press Enter then move ")
        buf.clear()
        collecting[0] = True
        for c in (3, 2, 1):
            print(f"\r  {clr(str(c), YELLOW)}...", end="", flush=True)
            await asyncio.sleep(0.35)
        print(f"\r  {clr('recording...', CYAN)}           ", end="", flush=True)
        await asyncio.sleep(2.0)
        collecting[0] = False
        if len(buf) >= 50:
            raw = np.array(buf, dtype=np.float32)
            recordings.setdefault(NULL_CLASS, []).append(raw)
            save_phrase_recordings(recordings, user_id)
            print(f"\r  {clr('saved', GREEN)}  null rep {len(recordings[NULL_CLASS])} ({len(buf)} samples)          ")
        else:
            print(f"\r  {clr('too short', YELLOW)} — skipped          ")

    client_task.cancel()
    print(clr(f"\n  Done — {len(recordings.get(NULL_CLASS, []))} total null reps saved.\n", BOLD))


async def _train_words_terminal(
    user_id: str,
    device_mac: str,
    reps: int = DTW_TRAIN_REPS,
) -> None:
    """
    Terminal training mode for DTW phrase recognition.
    For each phrase: countdown, perform the sign, auto-detect end by RMS drop.
    """
    from myo import MyoClient
    from myo.types import EMGData, IMUData, EMGMode, IMUMode, ClassifierMode

    print(clr(f"\n  Phrase training — user '{user_id}'  ({reps} reps per phrase)", BOLD))
    print(clr("  Perform each sign naturally. Recording stops automatically when\n"
              "  your hand drops to rest. Press Enter to start each rep.\n", DIM))

    imu_state     = [np.zeros(9, dtype=np.float32)]
    gesture_buf: list  = []
    recent_buf:  list  = []   # rolling window used for onset detection
    waiting_onset = [False]   # True = watching for RMS spike to start gesture_buf
    recording     = [False]   # True = gesture_buf is filling
    quiet_count   = [0]
    done          = [False]

    class WordsMyo(MyoClient):
        async def on_emg_data(self, emg: EMGData) -> None:
            for sample in (emg.sample1, emg.sample2):
                row      = np.array(list(sample), dtype=np.float32)
                full_row = np.concatenate([row, imu_state[0]])

                # --- Phase 1: wait for onset (same threshold as live inference) ---
                if waiting_onset[0]:
                    recent_buf.append(full_row)
                    if len(recent_buf) > DTW_RMS_WINDOW:
                        recent_buf.pop(0)
                    rms = float(np.sqrt(np.mean(np.array([r[:8] for r in recent_buf]) ** 2)))
                    if rms >= DTW_ONSET_RMS:
                        waiting_onset[0] = False
                        recording[0]     = True
                        gesture_buf.clear()
                        gesture_buf.append(full_row)  # start from onset frame, matching inference
                    continue

                # --- Phase 2: record until RMS drops ---
                if not recording[0]:
                    continue

                gesture_buf.append(full_row)
                window = gesture_buf[-DTW_RMS_WINDOW:]
                rms = float(np.sqrt(np.mean(np.array([r[:8] for r in window]) ** 2)))
                if rms < DTW_OFFSET_RMS:
                    quiet_count[0] += 1
                else:
                    quiet_count[0] = 0

                if (
                    len(gesture_buf) >= DTW_MIN_GESTURE
                    and quiet_count[0] >= DTW_MIN_QUIET
                ) or len(gesture_buf) >= DTW_MAX_GESTURE:
                    recording[0] = False
                    done[0]      = True

        async def on_imu_data(self, imu: IMUData) -> None:
            try:
                o = imu.orientation; g = imu.gyroscope; a = imu.accelerometer
                imu_state[0] = np.array(
                    [o.w, o.x, o.y, g[0], g[1], g[2], a[0], a[1], a[2]],
                    dtype=np.float32,
                )
            except Exception: pass

        async def on_emg_data_aggregated(self, eds) -> None: pass
        async def on_aggregated_data(self, ad) -> None: pass
        async def on_fv_data(self, fvd) -> None: pass
        async def on_motion_event(self, me) -> None: pass
        async def on_classifier_event(self, ce) -> None: pass

    print(clr("  Connecting to Myo...", CYAN))
    client = await WordsMyo.with_device(mac=device_mac or None)
    await client.setup(
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_ALL,
        classifier_mode=ClassifierMode.DISABLED,
    )
    print(clr("  Connected.\n", GREEN))

    loop = asyncio.get_event_loop()

    # Load recordings accumulated from previous sessions
    phrase_recordings: dict[str, list] = {
        p: list(recs) for p, recs in load_phrase_recordings(user_id).items()
    }
    existing_total = sum(len(v) for v in phrase_recordings.values())
    if existing_total:
        print(clr(f"  Loaded {existing_total} existing recordings "
                  f"({len(phrase_recordings)} phrases) from previous sessions.", CYAN))
        for p, recs in sorted(phrase_recordings.items()):
            print(f"    {p}: {len(recs)} reps")
        print(clr("  New reps will be added on top. "
                  f"Delete models/user_{user_id}/phrase_recordings.pkl to start fresh.\n", DIM))
    else:
        print(clr("  No existing recordings found — starting fresh.\n", DIM))

    client_task = asyncio.create_task(client.start())
    await asyncio.sleep(1.0)

    for phrase in PHRASES:
        print(f"  {clr(phrase.upper(), BOLD + CYAN)}")
        rep = 0
        while rep < reps:
            await loop.run_in_executor(None, input,
                f"    Rep {rep+1}/{reps} — ready? Press Enter then perform the sign ")

            # 3-2-1 countdown — gives user time to prepare
            for c in (3, 2, 1):
                print(f"\r    {clr(str(c), YELLOW)}...", end="", flush=True)
                await asyncio.sleep(0.35)
            print(f"\r    {clr('waiting...', CYAN)}  start signing when ready          ",
                  end="", flush=True)

            recent_buf.clear()
            gesture_buf.clear()
            quiet_count[0]  = 0
            done[0]         = False
            waiting_onset[0] = True  # on_emg_data waits for onset, then flips recording[0]

            # Wait for onset + auto-stop (or hard timeout)
            timeout = 10.0
            t0      = asyncio.get_event_loop().time()
            while not done[0] and (asyncio.get_event_loop().time() - t0) < timeout:
                if recording[0]:
                    frames = len(gesture_buf)
                    print(f"\r    {clr('* REC', RED)}  {frames} fr    ", end="", flush=True)
                await asyncio.sleep(0.02)
            waiting_onset[0] = False
            recording[0]     = False

            n_frames = len(gesture_buf)
            if n_frames < DTW_MIN_GESTURE:
                print(clr(f"\r    too short ({n_frames} samples) — try again", RED))
                continue  # don't advance rep

            raw = np.array(gesture_buf, dtype=np.float32)
            phrase_recordings.setdefault(phrase, []).append(raw)
            rep += 1
            print(clr(f"\r    {n_frames} samples recorded ({rep}/{reps})    ", GREEN))

        print()

    client_task.cancel()
    try: await client_task
    except (asyncio.CancelledError, Exception): pass
    await client.stop()
    await client.disconnect()

    # Persist raw recordings first — safe to interrupt training after this point
    save_phrase_recordings(phrase_recordings, user_id)
    total_reps = sum(len(v) for v in phrase_recordings.values())
    print(clr(f"\n  Saved {total_reps} total recordings "
              f"({total_reps - existing_total} new + {existing_total} from previous sessions).", CYAN))

    print(clr("  Training SVM model...", CYAN))
    model = train_dtw(phrase_recordings)
    save_dtw_model(model, user_id)
    print(clr(f"  Done! Run without --train-words to start recognition.", GREEN))
    print(clr(f"  Run --train-words again any time to add more reps and improve accuracy.\n", DIM))


async def main(args: argparse.Namespace) -> None:
    print()
    print(clr("  Echo  |  ASL Recognition  |  DyFAV + DTW", BOLD))
    print(clr("  Myo BLE → EMG+IMU → letters / phrases → LLM → English", DIM))
    print()

    if args.scan:
        await scan_devices()
        return

    if args.inspect:
        await _inspect(device_mac=args.device)
        return

    if args.train_null:
        await _train_null_terminal(
            user_id=args.user,
            device_mac=args.device,
            reps=args.train_null_reps,
        )
        return

    if args.train_words:
        await _train_words_terminal(
            user_id=args.user,
            device_mac=args.device,
            reps=args.train_words_reps,
        )
        return

    if args.train:
        await _train_terminal(
            user_id=args.user,
            device_mac=args.device,
            reps=args.train_reps,
        )
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
    parser.add_argument("--device",    default="",        help="Myo MAC address (optional; auto-discovers if omitted)")
    parser.add_argument("--user",      default="default",  help="User ID for model loading/saving")
    parser.add_argument("--scan",      action="store_true", help="Scan for BLE devices and exit")
    parser.add_argument("--inspect",   action="store_true", help="Stream raw EMG+IMU values to terminal (for verifying sensor layout)")
    parser.add_argument("--train",       action="store_true", help="Terminal training mode — record ASL letters (DyFAV)")
    parser.add_argument("--train-reps",  type=int, default=TRAIN_REPS_NEEDED, dest="train_reps",
                        help=f"Recordings per letter in --train mode (default: {TRAIN_REPS_NEEDED})")
    parser.add_argument("--train-words", action="store_true", dest="train_words",
                        help="Terminal training mode — record phrases/words (DTW)")
    parser.add_argument("--train-words-reps", type=int, default=DTW_TRAIN_REPS, dest="train_words_reps",
                        help=f"Recordings per phrase in --train-words mode (default: {DTW_TRAIN_REPS})")
    parser.add_argument("--train-null", action="store_true", dest="train_null",
                        help="Terminal training mode — record null/background gestures")
    parser.add_argument("--train-null-reps", type=int, default=30, dest="train_null_reps",
                        help="Number of null reps to record (default: 30)")
    parser.add_argument("--no-llm",     action="store_true", help="Skip LLM sentence reconstruction")
    parser.add_argument("--ws-port",  type=int, default=8765, dest="ws_port",
                        help="WebSocket server port (default: 8765)")
    asyncio.run(main(parser.parse_args()))

"""
WebSocket stream handler for real-time EMG-ASL inference.

Endpoint : ``/stream``
Protocol : Binary WebSocket frames containing raw BLE bytes (8-channel
           interleaved int16, little-endian).
Response : JSON text frames — ``{"label": "A", "confidence": 0.94, "timestamp_ms": 12345}``

Debounce
--------
Predictions are only emitted when:
  1. ``confidence >= CONFIDENCE_THRESHOLD``
  2. At least ``DEBOUNCE_MS`` milliseconds have elapsed since the last emission.

This prevents the client from being flooded with repeated labels for the
same sustained gesture.

Multiple simultaneous WebSocket clients are fully supported; each connection
gets its own ``EMGPipeline`` instance and inference state so that different
users / devices do not interfere with each other.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("emg_asl.ws")

router = APIRouter()


# ---------------------------------------------------------------------------
# Inline imports — avoids circular import with main.py at module load time.
# ---------------------------------------------------------------------------


def _get_app_state() -> Any:
    from .main import app_state  # noqa: PLC0415

    return app_state


def _get_constants() -> tuple[float, int]:
    from ..utils.constants import CONFIDENCE_THRESHOLD, DEBOUNCE_MS  # noqa: PLC0415

    return CONFIDENCE_THRESHOLD, DEBOUNCE_MS


# ---------------------------------------------------------------------------
# EMGStreamHandler
# ---------------------------------------------------------------------------


class EMGStreamHandler:
    """Per-connection handler that owns the pipeline and inference state.

    Parameters
    ----------
    websocket:
        The active WebSocket connection for this client.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self._app_state = _get_app_state()
        self._confidence_threshold, self._debounce_ms = _get_constants()

        # Each connection gets its own pipeline instance so calibration state
        # and ring-buffer contents do not bleed between users.
        from ..utils.pipeline import EMGPipeline  # noqa: PLC0415

        self._pipeline = EMGPipeline()

        # Debounce state.
        self._last_emit_time_ms: float = 0.0
        self._last_label: str = ""

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_inference(self, features: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence) for the given feature vector.

        Uses ONNX runtime when available, otherwise falls back to PyTorch.
        """
        state = self._app_state

        if state.use_onnx and state.ort_session is not None:
            input_name = state.ort_session.get_inputs()[0].name
            feed = {input_name: features[np.newaxis, :].astype(np.float32)}
            logits = state.ort_session.run(None, feed)[0][0]  # (num_classes,)
            probs = _softmax(logits)
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx])

            from ..utils.constants import ASL_LABELS  # noqa: PLC0415

            label = (
                ASL_LABELS[class_idx]
                if class_idx < len(ASL_LABELS)
                else str(class_idx)
            )
            return label, confidence

        elif state.torch_model is not None:
            return state.torch_model.predict(features)

        else:
            logger.warning("No inference backend available — returning null prediction.")
            return "UNKNOWN", 0.0

    def _should_emit(self, label: str, confidence: float) -> bool:
        """Return True when the debounce interval has elapsed and confidence is sufficient."""
        if confidence < self._confidence_threshold:
            return False
        now_ms = time.monotonic() * 1000.0
        elapsed_ms = now_ms - self._last_emit_time_ms
        if elapsed_ms < self._debounce_ms:
            return False
        return True

    def _record_emission(self) -> float:
        """Update the last-emit timestamp and return current time in ms."""
        now_ms = time.monotonic() * 1000.0
        self._last_emit_time_ms = now_ms
        return now_ms

    # ------------------------------------------------------------------
    # Main receive-process-send loop
    # ------------------------------------------------------------------

    async def handle(self) -> None:
        """Drive the entire lifecycle for one WebSocket connection.

        Receives binary frames, feeds them through the pipeline, runs
        inference, and streams JSON results back to the client.
        """
        client = self.ws.client
        logger.info("WebSocket connected: %s", client)

        try:
            while True:
                try:
                    raw_bytes: bytes = await self.ws.receive_bytes()
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected gracefully: %s", client)
                    break

                # Ingest raw BLE payload into the pipeline.
                n_decoded = self._pipeline.ingest_bytes(raw_bytes)
                if n_decoded == 0:
                    continue

                # Drain all available windows synchronously.
                # Each window → feature extraction → inference → optional emit.
                # Wrap in executor so CPU-bound work doesn't block the event loop.
                await asyncio.get_event_loop().run_in_executor(
                    None, self._drain_windows
                )

                # Send queued messages if any were produced.
                if self._pending_messages:
                    for msg in self._pending_messages:
                        await self.ws.send_text(msg)
                    self._pending_messages.clear()

        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error in WebSocket handler: %s", exc, exc_info=True)
        finally:
            self._pipeline.reset()
            logger.info("WebSocket session ended: %s", client)

    def _drain_windows(self) -> None:
        """Pull all ready windows from the pipeline and append JSON to ``_pending_messages``."""
        self._pending_messages: list[str] = []

        while True:
            window = self._pipeline.get_next_window()
            if window is None:
                break

            try:
                features = self._pipeline.process_window(window)
                label, confidence = self._run_inference(features)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Inference error: %s", exc)
                continue

            if self._should_emit(label, confidence):
                now_ms = self._record_emission()
                self._last_label = label
                payload = json.dumps(
                    {
                        "label": label,
                        "confidence": round(confidence, 4),
                        "timestamp_ms": round(now_ms),
                    }
                )
                self._pending_messages.append(payload)
                logger.debug("Emit: %s", payload)


# ---------------------------------------------------------------------------
# Softmax (avoid importing torch just for this)
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# FastAPI route
# ---------------------------------------------------------------------------


@router.websocket("/stream")
async def stream_endpoint(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and hand it to ``EMGStreamHandler``."""
    await websocket.accept()
    handler = EMGStreamHandler(websocket)
    await handler.handle()

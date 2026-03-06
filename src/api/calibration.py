"""
FastAPI router for user-specific EMG-ASL calibration.

Endpoints
---------
POST   /calibrate/start                   — Create a new calibration session.
POST   /calibrate/sample                  — Submit a labeled feature window.
POST   /calibrate/finish                  — Fine-tune the model, save profile.
GET    /calibrate/profile/{user_id}       — Load / inspect a saved profile.
DELETE /calibrate/profile/{user_id}       — Delete a user profile.

Session lifecycle
-----------------
1. ``POST /calibrate/start``   → returns ``{"session_id": "...", "user_id": "..."}``
2. Repeat ``POST /calibrate/sample`` for each labeled window.
3. ``POST /calibrate/finish``  → runs ``UserCalibrator.calibrate()``, saves profile,
   returns accuracy stats.
4. The inference WebSocket (``/stream``) automatically uses the last-loaded
   user profile when available in ``app_state``.

Sessions are stored in an in-process dictionary keyed by ``session_id``.
In a multi-process deployment you would replace this with Redis or another
shared store.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("emg_asl.calibration")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class StartCalibrationRequest(BaseModel):
    """Body for ``POST /calibrate/start``."""

    user_id: str = Field(
        ...,
        description="Unique identifier for the user. May contain alphanumeric characters and hyphens.",
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]+$",
    )


class StartCalibrationResponse(BaseModel):
    session_id: str
    user_id: str
    message: str


class AddSampleRequest(BaseModel):
    """Body for ``POST /calibrate/sample``."""

    session_id: str = Field(..., description="Session ID returned by /calibrate/start.")
    label: str = Field(
        ...,
        description="ASL label for this window (e.g. 'A', 'HELLO').",
        min_length=1,
        max_length=32,
    )
    features: list[float] = Field(
        ...,
        description="Flat feature vector produced by EMGPipeline.process_window().",
        min_length=1,
    )


class AddSampleResponse(BaseModel):
    session_id: str
    n_samples_total: int
    samples_per_class: dict[str, int]


class FinishCalibrationRequest(BaseModel):
    """Body for ``POST /calibrate/finish``."""

    session_id: str = Field(..., description="Session ID to finalise.")


class CalibrationStats(BaseModel):
    n_total: int
    n_classes_seen: int
    samples_per_class: dict[str, int]
    estimated_accuracy: float | None


class FinishCalibrationResponse(BaseModel):
    session_id: str
    user_id: str
    profile_path: str
    stats: CalibrationStats
    message: str


class ProfileResponse(BaseModel):
    user_id: str
    profile_dir: str
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# In-process session store
# ---------------------------------------------------------------------------

# session_id -> {"user_id": str, "calibrator": UserCalibrator}
_sessions: dict[str, dict[str, Any]] = {}


def _get_app_state() -> Any:
    from .main import app_state  # noqa: PLC0415

    return app_state


def _get_profile_dir() -> str:
    return _get_app_state().profile_dir


def _get_session(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Calibration session '{session_id}' not found or already finalised.",
        )
    return session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/start", response_model=StartCalibrationResponse)
async def start_calibration(body: StartCalibrationRequest) -> StartCalibrationResponse:
    """Create a new calibration session for *user_id*.

    Returns a ``session_id`` that must be included in all subsequent requests
    for this calibration run.
    """
    from ..models.calibration import UserCalibrator  # noqa: PLC0415
    from ..utils.constants import ASL_LABELS  # noqa: PLC0415

    session_id = str(uuid.uuid4())
    calibrator = UserCalibrator(label_names=ASL_LABELS)

    _sessions[session_id] = {
        "user_id": body.user_id,
        "calibrator": calibrator,
    }

    logger.info(
        "Calibration session started: session_id=%s user_id=%s",
        session_id,
        body.user_id,
    )
    return StartCalibrationResponse(
        session_id=session_id,
        user_id=body.user_id,
        message=f"Calibration session created. Submit labeled samples to /calibrate/sample.",
    )


@router.post("/sample", response_model=AddSampleResponse)
async def add_sample(body: AddSampleRequest) -> AddSampleResponse:
    """Append a single labeled feature window to the calibration session.

    The ``features`` list should be the output of
    ``EMGPipeline.process_window(window)`` — a flat 1-D vector.
    """
    session = _get_session(body.session_id)
    calibrator = session["calibrator"]

    features_np = np.array(body.features, dtype=np.float32)

    try:
        calibrator.add_sample(features_np, body.label)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    stats = calibrator.get_stats()
    logger.debug(
        "Sample added: session_id=%s label=%s total=%d",
        body.session_id,
        body.label,
        stats["n_total"],
    )
    return AddSampleResponse(
        session_id=body.session_id,
        n_samples_total=stats["n_total"],
        samples_per_class=stats["samples_per_class"],
    )


@router.post("/finish", response_model=FinishCalibrationResponse)
async def finish_calibration(body: FinishCalibrationRequest) -> FinishCalibrationResponse:
    """Fine-tune the model on the accumulated samples and save the user profile.

    This is the most computationally expensive step (50 epochs of FC fine-tuning).
    On a CPU this typically completes in < 5 seconds for ≤ 200 samples.

    After completion the session is removed from the in-process store.
    The inference WebSocket will automatically use the profile on subsequent
    connections if the client requests it via the ``user_id`` query parameter.
    """
    session = _get_session(body.session_id)
    user_id: str = session["user_id"]
    calibrator = session["calibrator"]

    # Retrieve the base model from app state.
    app_state = _get_app_state()
    if app_state.torch_model is None:
        raise HTTPException(
            status_code=503,
            detail="No PyTorch model loaded. Cannot perform calibration.",
        )

    try:
        fine_tuned = calibrator.calibrate(app_state.torch_model)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    profile_dir = _get_profile_dir()
    try:
        model_path = calibrator.save_profile(user_id, fine_tuned, profile_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save profile: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Profile save failed: {exc}"
        ) from exc

    stats_raw = calibrator.get_stats()
    stats = CalibrationStats(
        n_total=stats_raw["n_total"],
        n_classes_seen=stats_raw["n_classes_seen"],
        samples_per_class=stats_raw["samples_per_class"],
        estimated_accuracy=stats_raw.get("estimated_accuracy"),
    )

    # Remove the session to free memory.
    del _sessions[body.session_id]

    logger.info(
        "Calibration finished: user_id=%s profile_path=%s accuracy=%s",
        user_id,
        model_path,
        stats.estimated_accuracy,
    )

    return FinishCalibrationResponse(
        session_id=body.session_id,
        user_id=user_id,
        profile_path=model_path,
        stats=stats,
        message=(
            f"Calibration complete. Profile saved. "
            f"LOO accuracy estimate: "
            f"{stats.estimated_accuracy:.1%}" if stats.estimated_accuracy is not None
            else "Calibration complete. Profile saved."
        ),
    )


@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str) -> ProfileResponse:
    """Return metadata for a saved user profile.

    The model is *not* loaded into the inference session by this endpoint;
    it only inspects the profile directory and returns stored metadata.
    To activate a profile for inference, the client should reconnect to
    ``/stream?user_id=<user_id>``.
    """
    profile_dir = Path(_get_profile_dir()) / user_id
    if not profile_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for user '{user_id}'.",
        )

    import json  # noqa: PLC0415

    meta_path = profile_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            meta: dict[str, Any] = json.load(fh)
    else:
        meta = {"user_id": user_id, "note": "meta.json missing"}

    return ProfileResponse(
        user_id=user_id,
        profile_dir=str(profile_dir),
        meta=meta,
    )


@router.delete("/profile/{user_id}")
async def delete_profile(user_id: str) -> dict[str, str]:
    """Permanently delete the user profile directory.

    All fine-tuned model weights and metadata for *user_id* are removed.
    """
    profile_dir = Path(_get_profile_dir()) / user_id
    if not profile_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for user '{user_id}'.",
        )

    try:
        shutil.rmtree(profile_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to delete profile for '%s': %s", user_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete profile: {exc}",
        ) from exc

    logger.info("Deleted profile for user '%s'", user_id)
    return {"message": f"Profile for user '{user_id}' deleted successfully."}

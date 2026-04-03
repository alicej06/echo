"""
EMG-ASL inference server -- FastAPI application entry point.

Responsibilities
----------------
* Create and share the ``EMGPipeline`` singleton across all requests.
* Load the inference backend (ONNX preferred, PyTorch fallback) at startup.
* Download the ONNX model from Cloudflare R2 at startup if not present locally
  and ``R2_MODEL_URL`` is set in the environment.
* Mount the WebSocket stream endpoint (``/stream``).
* Include the calibration REST router (``/calibrate/...``).
* Configure CORS so the companion mobile app can reach all endpoints.
* Emit structured JSON logs throughout.

Running
-------
::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Or use the ``run()`` helper at the bottom for programmatic startup.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging — structured JSON-style output compatible with log aggregators.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("emg_asl.api")


# ---------------------------------------------------------------------------
# Import project modules after logging is configured so their loggers inherit.
# ---------------------------------------------------------------------------

from ..utils.constants import (
    ONNX_MODEL_PATH,
    PYTORCH_MODEL_PATH,
    PROFILE_DIR,
    REST_PORT,
    WS_HOST,
    WS_PORT,
)
from ..utils.pipeline import EMGPipeline
from ..models.lstm_classifier import ASLEMGClassifier
from ..utils.constants import ASL_LABELS, FEATURE_VECTOR_SIZE, NUM_CLASSES, SAMPLE_RATE

# Lazy-imported so the server starts even if onnxruntime isn't installed.
_ort: Any = None


# ---------------------------------------------------------------------------
# Application state — shared across the whole process lifetime.
# ---------------------------------------------------------------------------


class AppState:
    """Container for objects that must persist across request lifetimes."""

    pipeline: EMGPipeline
    torch_model: ASLEMGClassifier | None
    ort_session: Any | None  # onnxruntime.InferenceSession
    use_onnx: bool
    profile_dir: str

    def __init__(self) -> None:
        self.pipeline = EMGPipeline()
        self.torch_model = None
        self.ort_session = None
        self.use_onnx = False
        self.profile_dir = PROFILE_DIR


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown logic executed once per process."""

    # --- Startup -------------------------------------------------------
    logger.info("Starting EMG-ASL inference server")

    # Ensure the profile directory exists.
    Path(app_state.profile_dir).mkdir(parents=True, exist_ok=True)

    # Download model from R2 if not present locally and R2_MODEL_URL is set.
    onnx_path = Path(os.getenv("ONNX_MODEL_PATH", "models/asl_emg_classifier.onnx"))
    if not onnx_path.exists():
        r2_url = os.getenv("R2_MODEL_URL", "")
        if r2_url:
            logger.info("Downloading model from R2: %s", r2_url)
            import httpx

            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            with httpx.stream("GET", r2_url) as resp:
                with open(onnx_path, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            logger.info("Model downloaded successfully")
        else:
            logger.warning("No model found and R2_MODEL_URL not set. Using random weights.")

    # Try loading the ONNX model first.
    onnx_path = Path(ONNX_MODEL_PATH)
    if onnx_path.exists():
        try:
            global _ort
            import onnxruntime as ort  # type: ignore[import]

            _ort = ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            app_state.ort_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            app_state.use_onnx = True
            logger.info("Loaded ONNX model from %s", onnx_path)
        except Exception as exc:
            logger.warning(
                "Failed to load ONNX model (%s); falling back to PyTorch.", exc
            )

    # If ONNX load failed or model not present, try PyTorch.
    if not app_state.use_onnx:
        pt_path = Path(PYTORCH_MODEL_PATH)
        if pt_path.exists():
            try:
                app_state.torch_model = ASLEMGClassifier.load(pt_path)
                logger.info("Loaded PyTorch model from %s", pt_path)
            except Exception as exc:
                logger.warning("Failed to load PyTorch model: %s", exc)
        else:
            # No model file at all — initialise a random model so the server
            # can still run (useful during development / calibration-only mode).
            logger.warning(
                "No pre-trained model found. Creating untrained model with random weights."
            )
            app_state.torch_model = ASLEMGClassifier(
                input_size=FEATURE_VECTOR_SIZE,
                num_classes=NUM_CLASSES,
                label_names=ASL_LABELS,
            )

    logger.info(
        "Inference backend: %s", "ONNX" if app_state.use_onnx else "PyTorch"
    )
    logger.info("EMG pipeline ready: %s", app_state.pipeline)

    yield  # ← server is running

    # --- Shutdown ------------------------------------------------------
    logger.info("Shutting down EMG-ASL inference server")
    app_state.pipeline.reset()
    if app_state.ort_session is not None:
        del app_state.ort_session
    if app_state.torch_model is not None:
        del app_state.torch_model
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="EMG-ASL Inference Server",
    description=(
        "Real-time American Sign Language recognition from 8-channel surface EMG. "
        "Provides a WebSocket stream endpoint for live inference and REST endpoints "
        "for user-specific model calibration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS — allow the React Native / Flutter mobile app (any origin during dev).
# ---------------------------------------------------------------------------

_cors_origins_raw = os.environ.get("CORS_ORIGINS", "*")
_cors_origins: list[str] = (
    ["*"] if _cors_origins_raw == "*" else _cors_origins_raw.split(",")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------

from .auth import APIKeyMiddleware, validate_api_key  # noqa: E402
from .calibration import router as calibration_router  # noqa: E402
from .websocket import router as ws_router  # noqa: E402

# API key middleware runs before every request.  Exempt paths (health, docs,
# WebSocket upgrades) are handled inside the middleware itself.
app.add_middleware(APIKeyMiddleware)

app.include_router(ws_router)
app.include_router(
    calibration_router,
    prefix="/calibrate",
    tags=["Calibration"],
    # All calibration routes require a valid API key.
    dependencies=[Depends(validate_api_key)],
)


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Meta"], dependencies=[Depends(validate_api_key)])
async def root() -> JSONResponse:
    """Server status and runtime information.

    Requires a valid ``X-API-Key`` header.  Use ``GET /health`` for
    unauthenticated liveness checks.
    """
    return JSONResponse(
        {
            "status": "ok",
            "inference_backend": "onnx" if app_state.use_onnx else "pytorch",
            "pipeline": repr(app_state.pipeline),
            "ws_endpoint": f"ws://{WS_HOST}:{WS_PORT}/stream",
        }
    )


@app.get("/health", tags=["Meta"])
async def health() -> JSONResponse:
    """Lightweight liveness probe for load balancers / container orchestrators.

    Returns ``model_loaded: true`` once either the ONNX session or a PyTorch
    model has been successfully loaded, allowing readiness-aware checks.
    """
    model_loaded: bool = app_state.use_onnx or app_state.torch_model is not None
    return JSONResponse({"status": "ok", "model_loaded": model_loaded})


@app.get("/info", tags=["Meta"])
async def info() -> JSONResponse:
    """Static metadata about the deployed model and signal-processing config."""
    return JSONResponse(
        {
            "version": "1.0.0",
            "classes": NUM_CLASSES,
            "sample_rate": SAMPLE_RATE,
            "labels": ASL_LABELS,
        }
    )


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------


def run(host: str = WS_HOST, port: int | None = None) -> None:
    """Launch the Uvicorn server from Python code (e.g. ``python -m src.api.main``).

    The port is resolved in this order:
    1. ``port`` argument (if explicitly provided by the caller).
    2. ``PORT`` environment variable (set by Railway and similar PaaS platforms).
    3. ``REST_PORT`` constant from ``constants.py`` (default: 8000).
    """
    resolved_port: int = port if port is not None else int(os.getenv("PORT", str(REST_PORT)))
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=resolved_port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run()

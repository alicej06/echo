"""
API key authentication for the EMG-ASL inference server.

Two complementary mechanisms are provided:

1. ``validate_api_key`` -- a FastAPI dependency (``Depends``) for individual
   route handlers.  Raises HTTP 401 if the presented key is not valid.

2. ``APIKeyMiddleware`` -- a Starlette ASGI middleware that enforces the same
   policy globally, before a request ever reaches a route handler.  Certain
   safe paths (health check, OpenAPI docs, WebSocket upgrades) are exempt so
   that load-balancer probes and browser doc views keep working without a key.

Environment variables
---------------------
MAIA_DISABLE_AUTH
    Set to ``"1"`` to bypass all key validation.  Intended for local
    development and Docker Compose environments where the network is already
    trusted.  Never set this in a production deployment.

MAIA_API_KEYS
    Comma-separated list of accepted API keys, e.g.::

        MAIA_API_KEYS=prod-key-abc123,another-key-xyz789

    When this variable is empty or absent AND auth is not disabled, a single
    hard-coded development key (``dev-key-change-me``) is used and a warning
    is logged on every server start.

Header
------
Clients must send::

    X-API-Key: <key>

with every authenticated request.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger("emg_asl.auth")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The HTTP header name clients must use to pass their key.
API_KEY_HEADER = "X-API-Key"

# Set this environment variable to "1" to disable all key enforcement.
DISABLE_AUTH_ENV = "MAIA_DISABLE_AUTH"

# Environment variable that holds the comma-separated list of valid keys.
_API_KEYS_ENV = "MAIA_API_KEYS"

# Fallback key used when MAIA_API_KEYS is not configured and auth is on.
_DEFAULT_DEV_KEY = "dev-key-change-me"

# Paths that are always reachable without a key (exact prefix match).
_EXEMPT_PREFIXES = (
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auth_disabled() -> bool:
    """Return True when the operator has opted out of key enforcement."""
    return os.environ.get(DISABLE_AUTH_ENV, "").strip() == "1"


def _valid_keys() -> frozenset[str]:
    """Return the set of currently accepted API keys.

    Reads ``MAIA_API_KEYS`` at call time so the server picks up changes
    without a restart (useful in development).  Production deployments
    typically set this once at container start.
    """
    raw = os.environ.get(_API_KEYS_ENV, "").strip()
    if raw:
        # Split on commas, strip surrounding whitespace from each key, and
        # discard any empty tokens produced by trailing commas.
        keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
        if keys:
            return keys

    # No keys configured -- fall back to the dev key and warn loudly.
    logger.warning(
        "MAIA_API_KEYS is not set. Accepting the default dev key '%s'. "
        "Set MAIA_API_KEYS in production or set MAIA_DISABLE_AUTH=1 for local dev.",
        _DEFAULT_DEV_KEY,
    )
    return frozenset({_DEFAULT_DEV_KEY})


def _key_is_valid(key: Optional[str]) -> bool:
    """Return True when *key* is present and in the accepted key set."""
    if not key:
        return False
    return key in _valid_keys()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def validate_api_key(
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """FastAPI dependency that enforces API key authentication on a single route.

    Usage::

        @app.post("/predict")
        async def predict(key: str = Depends(validate_api_key)):
            ...

    Returns
    -------
    str
        The validated API key (useful for logging or per-key rate limiting).

    Raises
    ------
    HTTPException(401)
        When the key is missing or does not match any accepted key.
    """
    if _auth_disabled():
        # Return a placeholder string so callers that type-annotate with str
        # still receive a non-None value.
        return api_key or "__auth_disabled__"

    if not _key_is_valid(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    return api_key  # type: ignore[return-value]  # guaranteed non-None by _key_is_valid


# ---------------------------------------------------------------------------
# Starlette middleware
# ---------------------------------------------------------------------------


class APIKeyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces API key authentication at the transport layer.

    Exempt paths and WebSocket upgrade requests are passed through without
    inspection so that:

    * ``GET /health`` works for load-balancer probes.
    * ``GET /docs``, ``GET /redoc``, ``GET /openapi.json`` work in browsers.
    * ``/stream`` (and any future WebSocket route) is excluded because
      WebSocket clients cannot send arbitrary HTTP headers in all environments;
      per-connection auth is handled inside the WebSocket handler itself.

    For all other HTTP requests the middleware checks for a valid
    ``X-API-Key`` header and returns a 401 JSON response if it is absent or
    unrecognised.

    Parameters
    ----------
    app:
        The ASGI application to wrap.  Passed automatically by FastAPI when
        you call ``app.add_middleware(APIKeyMiddleware)``.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # --- WebSocket upgrade requests ------------------------------------
        # The ``connection`` header value is "upgrade" for WebSocket handshakes.
        # We skip auth here; individual WebSocket handlers can enforce their own
        # key check after accepting the connection.
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # --- Exempt health and documentation paths ------------------------
        path = request.url.path
        for prefix in _EXEMPT_PREFIXES:
            if path == prefix or path.startswith(prefix + "/"):
                return await call_next(request)

        # --- Auth disabled at the environment level -----------------------
        if _auth_disabled():
            return await call_next(request)

        # --- Validate the key ---------------------------------------------
        presented_key = request.headers.get(API_KEY_HEADER)
        if not _key_is_valid(presented_key):
            logger.warning(
                "Rejected request from %s %s -- missing or invalid API key",
                request.method,
                path,
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid API key"},
            )

        return await call_next(request)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# Compiles all C-extension wheels (scipy, numpy, etc.) in an isolated layer
# so the final image never needs a compiler toolchain at runtime.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build-time system deps needed to compile scipy / numpy / tsfresh native exts
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        cmake \
        libatlas-base-dev \
        libopenblas-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install runtime deps only — dev/test deps live in requirements-dev.txt which
# we deliberately do not copy here.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# Thin, non-root image that contains only what the inference server needs.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# curl is needed for the HEALTHCHECK probe
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Application source
COPY src/ ./src/

# models/ is volume-mounted at runtime; copy the (possibly empty) directory
# so the path always exists inside the container.
COPY models/ ./models/

# ── Security: non-root user ───────────────────────────────────────────────────
RUN groupadd --system emg && \
    useradd --system --gid emg --no-create-home emg && \
    chown -R emg:emg /app

USER emg

# ── Networking ───────────────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 8765

# ── Runtime environment (constants.py reads these via os.environ.get) ────────
ENV WS_HOST=0.0.0.0
ENV WS_PORT=8765
ENV REST_PORT=8000
ENV MODEL_DIR=/app/models

# ── Health probe ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default entrypoint: production uvicorn (4 workers, no reload) ─────────────
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]

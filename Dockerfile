# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

# uv — fast Python package installer (copied from official image, no curl needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ffmpeg — required for LiveKit audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first so Docker layer cache reuses installs
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install project + dependencies (system-wide, no venv needed in container)
RUN uv pip install --system --no-cache -e .

# ── Runtime env defaults (override via Railway env vars) ──────────────────────
ENV PYTHONUNBUFFERED=1
ENV STT_PROVIDER=groq
ENV STT_MODEL=whisper-large-v3-turbo
ENV LLM_MODEL=llama-3.3-70b-versatile
ENV STT_LANGUAGE=en
ENV PIPER_MODEL_PATH=models/en_US-lessac-high.onnx
ENV MIN_ENDPOINTING_DELAY=0.25
ENV MAX_ENDPOINTING_DELAY=1.5
ENV ALLOW_INTERRUPTIONS=true

CMD ["python", "src/livekit_voice_agent/agent.py", "start"]

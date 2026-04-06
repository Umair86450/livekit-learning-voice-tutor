#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"
mkdir -p "$MODEL_DIR/whisper"

# English voice: Lessac (high quality)
EN_MODEL="en_US-lessac-high"
EN_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high"

if [ ! -f "$MODEL_DIR/${EN_MODEL}.onnx" ]; then
    echo "Downloading English Piper voice: $EN_MODEL ..."
    curl -L -o "$MODEL_DIR/${EN_MODEL}.onnx" "${EN_URL}/${EN_MODEL}.onnx"
    curl -L -o "$MODEL_DIR/${EN_MODEL}.onnx.json" "${EN_URL}/${EN_MODEL}.onnx.json"
    echo "English voice downloaded."
else
    echo "English voice already exists, skipping."
fi

# ── Whisper STT model (faster-whisper) ──────────────────────────────
# Reads LOCAL_STT_MODEL from .env — default: base
# Available: tiny | base | small | medium | large-v3

WHISPER_MODEL="${LOCAL_STT_MODEL:-base}"
WHISPER_DIR="$MODEL_DIR/whisper"

if [ ! -d "$WHISPER_DIR/$WHISPER_MODEL" ]; then
    echo "Downloading Whisper STT model: $WHISPER_MODEL (to $WHISPER_DIR) ..."
    uv run python - <<PYEOF
from faster_whisper import WhisperModel
print(f"  Downloading faster-whisper/$WHISPER_MODEL ...")
WhisperModel("$WHISPER_MODEL", device="cpu", compute_type="int8", download_root="$WHISPER_DIR")
print("  Whisper model downloaded.")
PYEOF
    echo "Whisper model saved to $WHISPER_DIR/$WHISPER_MODEL"
else
    echo "Whisper model '$WHISPER_MODEL' already exists at $WHISPER_DIR/$WHISPER_MODEL, skipping."
fi

echo "Done. Models saved to $MODEL_DIR/"

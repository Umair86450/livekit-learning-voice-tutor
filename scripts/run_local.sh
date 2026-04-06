#!/usr/bin/env bash
set -euo pipefail

echo "Starting LiveKit server (Docker) ..."
docker run -d --name livekit-dev \
    -p 7880:7880 \
    -p 7881:7881 \
    -p 7882:7882/udp \
    -e LIVEKIT_KEYS="devkey: secret" \
    livekit/livekit-server --dev \
    2>/dev/null || echo "LiveKit server already running"

echo "Starting voice agent ..."
uv run python src/livekit_voice_agent/agent.py dev

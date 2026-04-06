#!/bin/bash
cd "$(dirname "$0")/.."

# livekit-server check, install if missing
if ! command -v livekit-server &> /dev/null; then
    echo "Installing livekit-server via brew..."
    brew install livekit
fi

echo ""
echo "Starting LiveKit server (bind 0.0.0.0)..."
livekit-server --dev --bind 0.0.0.0 &
LIVEKIT_PID=$!
sleep 2

echo "Starting demo HTTP server on port 8080..."
uv run python demo/server.py &
SERVER_PID=$!
sleep 1

# Show local IP for same-network client demo (macOS)
DEMO_IP=""
if command -v ipconfig &> /dev/null; then
  DEMO_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
fi
if [ -z "$DEMO_IP" ] && command -v hostname &> /dev/null; then
  DEMO_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi

echo ""
echo "=============================================="
echo "  DEMO READY"
echo "=============================================="
echo "  1. In another terminal run the agent:"
echo "     uv run python src/livekit_voice_agent/agent.py dev"
echo ""
echo "  2. On this machine open:  http://localhost:8080"
if [ -n "$DEMO_IP" ]; then
  echo ""
  echo "  3. Client on same WiFi: open in browser:"
  echo "     http://${DEMO_IP}:8080"
  echo "     (Share this URL for client demo)"
fi
echo "=============================================="
echo ""

open http://localhost:8080 2>/dev/null || true

trap "kill $LIVEKIT_PID $SERVER_PID 2>/dev/null; exit" INT
wait

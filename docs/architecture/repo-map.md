# Repo Map

## Core Runtime
- `src/livekit_voice_agent/agent.py`: Agent session wiring, STT/LLM/TTS pipeline, RAG tools.
- `src/livekit_voice_agent/config.py`: `Settings` model and environment contract.
- `src/livekit_voice_agent/prompts.py`: Assistant system prompt and greeting.
- `src/livekit_voice_agent/rag.py`: Retrieval logic and tool-facing context assembly.

## Supporting Modules
- `src/livekit_voice_agent/stt.py`: Local whisper adapter.
- `src/livekit_voice_agent/tts.py`: Piper TTS adapter.
- `src/livekit_voice_agent/debug.py`: Latency/event logging hooks.
- `src/livekit_voice_agent/health.py`: Health monitoring and metrics.

## Validation
- `tests/`: Unit/regression checks by feature area.
- `scripts/`: Model download, RAG preparation, scraping, local run helpers.

# Commands Runbook

## Setup
- `uv sync`
- `bash scripts/download_models.sh`

## Run
- `uv run python src/livekit_voice_agent/agent.py console`
- `uv run python src/livekit_voice_agent/agent.py dev`

## Tests / Lint
- `uv run pytest -v`
- `uv run pytest tests/test_<changed_area>.py -v`
- `uv run ruff check .`

## Deploy
- `docker build -t livekit-voice-agent .`
- `docker run --rm --env-file .env livekit-voice-agent`
- Railway start: `python src/livekit_voice_agent/agent.py start`

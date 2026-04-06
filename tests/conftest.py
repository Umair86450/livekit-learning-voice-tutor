from __future__ import annotations

import pytest

from livekit_voice_agent.config import Settings


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "STT_PROVIDER",
        "STT_MODEL",
        "LLM_MODEL",
        "PIPER_MODEL_PATH",
        "LOCAL_STT_MODEL",
        "LOCAL_STT_DEVICE",
        "LOCAL_STT_COMPUTE_TYPE",
        "ALLOW_INTERRUPTIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    return Settings(_env_file=None, groq_api_key="test-key-123")

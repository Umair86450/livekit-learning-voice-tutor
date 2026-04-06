from __future__ import annotations

import pytest

from livekit_voice_agent.config import Settings


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "my-key")
    monkeypatch.setenv("STT_MODEL", "custom-model")
    s = Settings(_env_file=None, groq_api_key="my-key", stt_model="custom-model")
    assert s.groq_api_key == "my-key"
    assert s.stt_model == "custom-model"


def test_settings_defaults(mock_settings: Settings) -> None:
    assert mock_settings.livekit_url == "ws://localhost:7880"
    assert mock_settings.stt_provider == "groq"
    assert mock_settings.stt_model == "whisper-large-v3-turbo"
    assert mock_settings.llm_model == "llama-3.3-70b-versatile"
    assert mock_settings.piper_model_path == "models/en_US-lessac-high.onnx"
    assert mock_settings.local_stt_model == "base"
    assert mock_settings.local_stt_device == "cpu"
    assert mock_settings.local_stt_compute_type == "int8"
    assert mock_settings.allow_interruptions is True


def test_settings_requires_groq_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(Exception):
        Settings(_env_file=None)

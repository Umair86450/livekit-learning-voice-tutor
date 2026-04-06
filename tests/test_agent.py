from __future__ import annotations

from livekit_voice_agent.agent import VoiceAssistant
from livekit_voice_agent.config import Settings


def test_voice_assistant_english_prompt() -> None:
    settings = Settings(groq_api_key="test-key")
    agent = VoiceAssistant(settings=settings)
    assert "voice agent" in agent.instructions or "White Box" in agent.instructions


def test_voice_assistant_no_markdown_instruction() -> None:
    settings = Settings(groq_api_key="test-key")
    agent = VoiceAssistant(settings=settings)
    assert "markdown" in agent.instructions.lower()

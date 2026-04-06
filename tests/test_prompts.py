from __future__ import annotations

from livekit_voice_agent.prompts import get_greeting, get_system_prompt


def test_system_prompt_exists() -> None:
    prompt = get_system_prompt()
    assert "voice agent" in prompt or "White Box" in prompt


def test_system_prompt_no_markdown_instruction() -> None:
    prompt = get_system_prompt()
    assert "markdown" in prompt.lower()


def test_greeting_exists() -> None:
    greeting = get_greeting()
    assert any(w in greeting.lower() for w in ["hello", "help", "thanks"])

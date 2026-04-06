from __future__ import annotations

import pytest

from livekit_voice_agent.tts import (
    PIPER_NUM_CHANNELS,
    PIPER_SAMPLE_RATE,
    PiperTTS,
    _chunk_text_for_tts,
)


def test_piper_tts_init() -> None:
    p = PiperTTS(model_path="models/en.onnx")
    assert p.sample_rate == PIPER_SAMPLE_RATE
    assert p.num_channels == PIPER_NUM_CHANNELS
    assert p.label == "piper-tts"


def test_piper_tts_capabilities() -> None:
    p = PiperTTS(model_path="models/en.onnx")
    assert p.capabilities.streaming is False


def test_piper_tts_lazy_loading() -> None:
    p = PiperTTS(model_path="models/en.onnx")
    assert p._voice is None


def test_chunk_text_first_chunk_small() -> None:
    # First chunk should be <= FIRST_CHUNK_MAX_CHARS for low ttfb
    text = "This is a long sentence that should be split so the first part is small."
    chunks = _chunk_text_for_tts(text)
    assert len(chunks) >= 2
    assert len(chunks[0]) <= 35
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_chunk_text_sentence_boundaries() -> None:
    text = "First sentence. Second one! Third?"
    chunks = _chunk_text_for_tts(text)
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "First" in joined and "Second" in joined and "Third" in joined


def test_chunk_text_empty() -> None:
    assert _chunk_text_for_tts("") == []
    assert _chunk_text_for_tts("   ") == []


@pytest.mark.asyncio
async def test_piper_tts_synthesize_returns_stream() -> None:
    p = PiperTTS(model_path="models/en.onnx")
    stream = p.synthesize("Hello world")
    assert stream.input_text == "Hello world"
    await stream.aclose()

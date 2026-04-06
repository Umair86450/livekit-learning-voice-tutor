from __future__ import annotations

import pytest

from livekit_voice_agent.stt import LocalWhisperSTT, _buffer_to_float32_pcm


def test_local_whisper_stt_init() -> None:
    stt = LocalWhisperSTT(model_size="tiny", device="cpu", compute_type="int8", language="en")
    assert stt.capabilities.streaming is False
    assert stt.capabilities.interim_results is False
    assert stt.label == "local-whisper-stt"


def test_local_whisper_stt_lazy_model() -> None:
    stt = LocalWhisperSTT(model_size="tiny", device="cpu", language="en")
    assert stt._model is None


def test_buffer_to_float32_pcm_empty() -> None:
    pcm, duration = _buffer_to_float32_pcm([])
    assert pcm == b""
    assert duration == 0.0


def test_buffer_to_float32_pcm_empty_list() -> None:
    pcm, duration = _buffer_to_float32_pcm([])
    assert pcm == b""
    assert duration == 0.0


@pytest.mark.asyncio
async def test_local_whisper_recognize_empty_buffer() -> None:
    """Empty buffer returns empty transcript event."""
    from livekit.agents import APIConnectOptions
    from livekit.agents.stt import SpeechEventType

    stt = LocalWhisperSTT(model_size="tiny", device="cpu", compute_type="int8", language="en")
    event = await stt._recognize_impl(
        [],
        language="en",
        conn_options=APIConnectOptions(),
    )
    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert len(event.alternatives) == 1
    assert event.alternatives[0].text == ""
    assert event.recognition_usage is not None
    assert event.recognition_usage.audio_duration == 0.0


@pytest.mark.asyncio
async def test_local_whisper_recognize_silence_buffer() -> None:
    """Buffer with silence returns event with zero duration or empty text."""
    import numpy as np

    from livekit.agents import APIConnectOptions
    from livekit.agents.stt import SpeechEventType

    # Minimal PCM: 0.1s at 16kHz mono float32 (stored as bytes)
    n_samples = 1600
    silence = np.zeros(n_samples, dtype=np.float32)
    # _buffer_to_float32_pcm expects frames with .data (s16), .sample_rate, .num_channels
    # Build a minimal mock: bytes from float32 for our helper (it does frombuffer int16 or we pass float32)
    # Our helper uses np.frombuffer(data, int16) so we need s16 bytes
    silence_s16 = (silence * 32768).astype(np.int16)
    frame = type("Frame", (), {"data": silence_s16.tobytes(), "sample_rate": 16000, "num_channels": 1})()

    stt = LocalWhisperSTT(model_size="tiny", device="cpu", compute_type="int8", language="en")
    event = await stt._recognize_impl(
        [frame],
        language="en",
        conn_options=APIConnectOptions(),
    )
    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.recognition_usage is not None
    assert event.recognition_usage.audio_duration > 0

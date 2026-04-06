"""
Local STT using faster-whisper (no API key). Use with StreamAdapter + VAD for voice pipeline.
Optimized for low latency: small model, int8 on CPU, lazy load.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from livekit.agents import stt as livekit_stt, utils as lk_utils

if TYPE_CHECKING:
    from livekit.agents.utils.audio import AudioBuffer

logger = logging.getLogger(__name__)

# Whisper expects 16 kHz mono float32
WHISPER_SAMPLE_RATE = 16000


def _buffer_to_float32_pcm(buffer: Any) -> tuple[bytes, int]:
    """Convert AudioBuffer to mono 16 kHz PCM bytes and original duration in seconds."""
    import numpy as np

    frames = buffer if isinstance(buffer, list) else [buffer]
    if not frames:
        return b"", 0.0

    chunks: list[np.ndarray] = []
    sample_rate = getattr(frames[0], "sample_rate", 16000)
    for f in frames:
        frame = f  # rtc.AudioFrame
        data = getattr(frame, "data", None) or b""
        if not data:
            continue
        # PCM s16le -> float32 [-1, 1]
        samples = np.frombuffer(data, dtype=np.int16)
        if getattr(frame, "num_channels", 1) == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        chunks.append(samples.astype(np.float32) / 32768.0)
        if hasattr(frame, "sample_rate"):
            sample_rate = frame.sample_rate

    if not chunks:
        return b"", 0.0
    audio = np.concatenate(chunks)
    duration_sec = len(audio) / sample_rate

    if sample_rate != WHISPER_SAMPLE_RATE:
        # Resample to 16 kHz
        n_out = int(len(audio) * WHISPER_SAMPLE_RATE / sample_rate)
        indices = np.linspace(0, len(audio) - 1, n_out, dtype=np.float32)
        audio = np.interp(indices, np.arange(len(audio), dtype=np.float32), audio)

    return audio.tobytes(), duration_sec


class LocalWhisperSTT(livekit_stt.STT):
    """Local speech-to-text using faster-whisper. No API key. Use with StreamAdapter + VAD."""

    def __init__(
        self,
        *,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        download_root: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=livekit_stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language or None
        self._download_root = download_root
        self._model: Any = None

    @property
    def label(self) -> str:
        return "local-whisper-stt"

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading local Whisper STT: model=%s device=%s compute_type=%s",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
            )
        return self._model

    async def _recognize_impl(
        self,
        buffer: "AudioBuffer",
        *,
        language: str | None,
        conn_options: Any,
    ) -> livekit_stt.SpeechEvent:
        import numpy as np

        from livekit.agents.stt import (
            RecognitionUsage,
            SpeechData,
            SpeechEvent,
            SpeechEventType,
        )

        pcm_bytes, duration_sec = _buffer_to_float32_pcm(buffer)
        if not pcm_bytes or duration_sec <= 0:
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                request_id=lk_utils.shortuuid(),
                alternatives=[SpeechData(language=language or self._language or "en", text="")],
                recognition_usage=RecognitionUsage(audio_duration=duration_sec),
            )

        lang = language or self._language or "en"
        model = self._ensure_model()
        audio_f32 = np.frombuffer(pcm_bytes, dtype=np.float32)

        def _transcribe() -> tuple[str, str]:
            segments, info = model.transcribe(audio_f32, language=lang if lang != "auto" else None)
            text = " ".join(s.text for s in segments).strip()
            detected = (info.language or "en") if hasattr(info, "language") else lang
            return text, detected

        text, detected_lang = await asyncio.to_thread(_transcribe)

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=lk_utils.shortuuid(),
            alternatives=[SpeechData(language=detected_lang, text=text)],
            recognition_usage=RecognitionUsage(audio_duration=duration_sec),
        )

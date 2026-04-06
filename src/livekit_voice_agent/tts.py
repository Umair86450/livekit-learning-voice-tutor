from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from livekit.agents import APIConnectOptions, tts, utils

DEFAULT_CONN_OPTIONS = APIConnectOptions()

logger = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22050
PIPER_NUM_CHANNELS = 1

# Streaming: first chunk small = low ttfb; rest up to MAX_CHUNK_CHARS
FIRST_CHUNK_MAX_CHARS = 35
MAX_CHUNK_CHARS = 60


def _chunk_text_for_tts(text: str) -> list[str]:
    """Split text into speakable chunks. First chunk kept small so first audio plays fast (low ttfb)."""
    text = text.strip()
    if not text:
        return []

    def split_by_size(s: str, max_len: int) -> list[str]:
        out: list[str] = []
        parts = re.split(r"(?<=[.!?])\s+", s)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_len:
                out.append(part)
                continue
            subparts = re.split(r"(?<=[,;])\s+", part)
            for sub in subparts:
                sub = sub.strip()
                if not sub:
                    continue
                if len(sub) <= max_len:
                    out.append(sub)
                    continue
                words = sub.split()
                current: list[str] = []
                current_len = 0
                for w in words:
                    need = len(w) + (1 if current else 0)
                    if current_len + need > max_len and current:
                        out.append(" ".join(current))
                        current, current_len = [], 0
                    current.append(w)
                    current_len += len(w) + (1 if len(current) > 1 else 0)
                if current:
                    out.append(" ".join(current))
        return out

    all_chunks = split_by_size(text, MAX_CHUNK_CHARS)
    if not all_chunks:
        return []
    # First chunk smaller for faster first-byte (real-time feel)
    first = all_chunks[0]
    if len(first) > FIRST_CHUNK_MAX_CHARS:
        words = first.split()
        head: list[str] = []
        n = 0
        for w in words:
            if n + len(w) + (1 if head else 0) > FIRST_CHUNK_MAX_CHARS and head:
                break
            head.append(w)
            n += len(w) + (1 if head else 0)
        rest_str = " ".join(words[len(head) :])
        first_chunk = " ".join(head)
        if rest_str.strip():
            rest_chunks = split_by_size(rest_str.strip(), MAX_CHUNK_CHARS)
            all_chunks = [first_chunk] + rest_chunks + all_chunks[1:]
        else:
            all_chunks = [first_chunk] + all_chunks[1:]
    return [c for c in all_chunks if c.strip()]


class PiperChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts_instance: PiperTTS,
        input_text: str,
        conn_options: APIConnectOptions = DEFAULT_CONN_OPTIONS,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._tts_instance = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=PIPER_SAMPLE_RATE,
            num_channels=PIPER_NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        voice = self._tts_instance._ensure_voice()
        chunks = _chunk_text_for_tts(self.input_text)
        if not chunks:
            output_emitter.flush()
            output_emitter.end_input()
            return

        logger.debug("TTS streaming %d chunks: %s...", len(chunks), self.input_text[:50])
        for chunk in chunks:
            raw_audio = await asyncio.to_thread(self._synthesize, voice, chunk)
            if raw_audio:
                output_emitter.push(raw_audio)
        output_emitter.flush()
        output_emitter.end_input()

    @staticmethod
    def _synthesize(voice: Any, text: str) -> bytes:
        import wave
        from io import BytesIO

        if not text.strip():
            return b""
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            return wf.readframes(wf.getnframes())


class PiperTTS(tts.TTS):
    def __init__(self, *, model_path: str) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=PIPER_SAMPLE_RATE,
            num_channels=PIPER_NUM_CHANNELS,
        )
        self._model_path = model_path
        self._voice: Any = None

    @property
    def label(self) -> str:
        return "piper-tts"

    def _ensure_voice(self) -> Any:
        if self._voice is None:
            from piper import PiperVoice

            logger.info("Loading Piper voice: %s", self._model_path)
            self._voice = PiperVoice.load(self._model_path)
        return self._voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_CONN_OPTIONS,
    ) -> PiperChunkedStream:
        return PiperChunkedStream(tts_instance=self, input_text=text, conn_options=conn_options)

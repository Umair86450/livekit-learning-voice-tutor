"""
Debug logging for the voice agent: STT, LLM, TTS, and latency breakdown.
All messages are in English for easier debugging.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from livekit.agents import AgentSession
from livekit.agents.metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics

logger = logging.getLogger("livekit_voice_agent.debug")

# Log section separators
_THICK = "=" * 64
_SEP = "-" * 64
_TARGET_V2V_MS = 700.0  # target voice-to-voice latency (ms)

# Stale turn TTL: if a turn doesn't complete within this time, it's dropped
TURN_TTL_SECONDS = 30.0


@dataclass
class _TurnData:
    """Per-turn metrics for one user utterance → agent response."""

    # User side
    user_audio_sec: float = 0.0
    stt_duration_ms: float = 0.0

    # Pipeline (user stopped talking → agent first audio)
    eou_ms: float = 0.0
    transcript_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    tts_ttfb_ms: float = 0.0

    # Agent side
    agent_audio_sec: float = 0.0
    agent_text: str = ""

    has_eou: bool = False
    has_llm: bool = False
    has_tts: bool = False

    # Timestamp for TTL-based memory leak prevention
    created_at: float = field(default_factory=time.monotonic)

    def voice_to_voice_ms(self) -> float:
        return self.eou_ms + self.transcript_ms + self.llm_ttft_ms + self.tts_ttfb_ms

    def is_complete(self) -> bool:
        return self.has_eou and self.has_llm and self.has_tts

    def is_stale(self, ttl: float = TURN_TTL_SECONDS) -> bool:
        return (time.monotonic() - self.created_at) > ttl


def _cleanup_stale_turns(turns: dict[str, _TurnData], ttl: float = TURN_TTL_SECONDS) -> int:
    """Remove turns older than ttl seconds. Returns count of removed entries."""
    stale = [sid for sid, t in turns.items() if t.is_stale(ttl)]
    for sid in stale:
        logger.warning("[DEBUG] Dropping stale turn %s (incomplete after %.0fs)", sid, ttl)
        turns.pop(sid)
    return len(stale)


def _bar(value: float, target: float, width: int = 24) -> str:
    """Visual bar (fixed width): █ = within budget, ▓ = over, ░ = unused. Length always = width."""
    if target <= 0:
        return "░" * width
    ratio = value / target
    if ratio <= 1.0:
        used = int(ratio * width)
        return "█" * used + "░" * (width - used)
    # Over budget: show full bar as "used" + one or a few "over" blocks, total = width
    used = width - 1
    return "█" * used + "▓"


def _fmt_sec(sec: float) -> str:
    if sec < 1:
        return f"{sec * 1000:.0f}ms"
    return f"{sec:.2f}s"


def _latency_summary(turn: _TurnData, turn_index: int) -> str:
    """Build the per-turn latency summary in English."""
    v2v = turn.voice_to_voice_ms()
    over = v2v - _TARGET_V2V_MS

    if v2v <= _TARGET_V2V_MS:
        verdict = f"FAST ({v2v:.0f}ms, under {_TARGET_V2V_MS:.0f}ms target)"
    elif v2v <= 1000:
        verdict = f"OK ({v2v:.0f}ms, +{over:.0f}ms over target)"
    elif v2v <= 2000:
        verdict = f"SLOW ({v2v:.0f}ms, +{over:.0f}ms over target)"
    else:
        verdict = f"VERY SLOW ({v2v:.0f}ms, +{over:.0f}ms over target)"

    budgets = [
        ("EOU wait      ", turn.eou_ms, 80, "Turn detector silence wait"),
        ("STT (transcript)", turn.transcript_ms, 70, "Speech-to-text completion"),
        ("LLM (ttft)    ", turn.llm_ttft_ms, 200, "LLM first token (Groq)"),
        ("TTS (ttfb)    ", turn.tts_ttfb_ms, 150, "First audio byte (Piper)"),
    ]

    worst = max(budgets, key=lambda x: x[1] - x[2])
    worst_name = worst[0].strip()
    worst_val = worst[1]
    worst_budget = worst[2]

    lines = [
        "",
        _THICK,
        f"  LATENCY SUMMARY — Turn #{turn_index}",
        _THICK,
        "",
        "  USER:",
        f"    Speech duration     : {_fmt_sec(turn.user_audio_sec)}",
        f"    STT processing time : {turn.stt_duration_ms:.0f}ms",
        "",
        "  AGENT:",
        f"    Response duration   : {_fmt_sec(turn.agent_audio_sec)}",
        "",
        "  VOICE-TO-VOICE (user stops → agent first audio):",
        f"    Total   : {v2v:.0f}ms   [{_bar(v2v, _TARGET_V2V_MS)}]  target={_TARGET_V2V_MS:.0f}ms",
        f"    Verdict : {verdict}",
        "",
        "  BREAKDOWN (where time was spent):",
    ]

    for name, val, budget, desc in budgets:
        diff = f"+{val - budget:.0f}ms over" if val > budget else f"{budget - val:.0f}ms under"
        lines.append(f"    {name}  {val:6.0f}ms  [{_bar(val, budget, 12)}]  budget={budget}ms  ({diff})  — {desc}")

    lines += [
        "",
        f"  BOTTLENECK : {worst_name}  ({worst_val:.0f}ms vs {worst_budget}ms budget)",
        "",
        _tip(worst_name, worst_val),
        _THICK,
    ]
    return "\n".join(lines)


def _tip(component: str, val_ms: float) -> str:
    """Suggestions in English based on the slowest component."""
    if "TTS" in component:
        return (
            "  TTS: Piper runs on CPU and synthesizes in chunks. "
            "Use shorter responses in system prompt, or tune Piper model/speed settings for lower ttfb."
        )
    if "EOU" in component:
        return (
            "  EOU: Turn detector waits for silence. Long user pauses increase this. "
            "preemptive_generation is already on; you can tune VAD min_silence_duration if needed."
        )
    if "LLM" in component:
        return (
            "  LLM: If timeouts occur, Groq may be rate-limited or overloaded. "
            "Try a smaller model (e.g. llama-3.1-8b-instant) or increase client timeout in plugin config."
        )
    return (
        "  STT: whisper-large-v3-turbo is already fast. "
        "Check network and Groq API status if STT is slow."
    )


def attach_debug_logging(session: AgentSession) -> None:
    """Attach structured debug logs: STT, LLM, TTS, and latency summary. All in English."""

    turns: dict[str, _TurnData] = {}
    turn_counter: list[int] = [0]  # mutable so inner fn can increment

    def _get(speech_id: str) -> _TurnData:
        if speech_id not in turns:
            turns[speech_id] = _TurnData()
        return turns[speech_id]

    def _maybe_summarize(speech_id: str | None) -> None:
        if not speech_id:
            return
        # Evict stale turns to prevent memory leak in long-running sessions
        _cleanup_stale_turns(turns)
        turn = turns.get(speech_id)
        if turn and turn.is_complete():
            turn_counter[0] += 1
            logger.info("%s", _latency_summary(turn, turn_counter[0]))
            turns.pop(speech_id, None)

    # ── STT: user speech → transcript ─────────────────────────────
    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:  # type: ignore[no-untyped-def]
        if ev.is_final:
            logger.info(
                "\n[STT] %s\n  USER said: %s\n%s",
                _SEP,
                ev.transcript,
                _SEP,
            )

    # ── LLM: agent reply text ───────────────────────────────────────
    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # type: ignore[no-untyped-def]
        item = ev.item
        if getattr(item, "role", None) == "assistant":
            text = _extract_text(item)
            if text:
                chars = len(text)
                logger.info(
                    "\n[LLM] %s\n  AGENT reply (%d chars): %s\n%s",
                    _SEP,
                    chars,
                    text,
                    _SEP,
                )

    # ── Metrics: STT / EOU / LLM / TTS timings ───────────────────────
    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:  # type: ignore[no-untyped-def]
        m = ev.metrics

        # Use getattr: some metric types / SDK versions may not have speech_id
        speech_id = getattr(m, "speech_id", None)

        if isinstance(m, STTMetrics):
            logger.info(
                "[STT]  processing_time=%.0fms  user_audio_duration=%s",
                m.duration * 1000,
                _fmt_sec(m.audio_duration),
            )
            if speech_id:
                t = _get(speech_id)
                t.user_audio_sec = m.audio_duration
                t.stt_duration_ms = m.duration * 1000

        elif isinstance(m, EOUMetrics):
            logger.info(
                "[EOU]  end_of_utterance_wait=%.0fms  transcript_delay=%.0fms",
                m.end_of_utterance_delay * 1000,
                m.transcription_delay * 1000,
            )
            if speech_id:
                t = _get(speech_id)
                t.eou_ms = m.end_of_utterance_delay * 1000
                t.transcript_ms = m.transcription_delay * 1000
                t.has_eou = True
                _maybe_summarize(speech_id)

        elif isinstance(m, LLMMetrics):
            logger.info(
                "[LLM]  ttft=%.0fms  total=%.0fms  completion_tokens=%d",
                m.ttft * 1000,
                m.duration * 1000,
                m.completion_tokens,
            )
            if speech_id:
                t = _get(speech_id)
                t.llm_ttft_ms = m.ttft * 1000
                t.has_llm = True
                _maybe_summarize(speech_id)

        elif isinstance(m, TTSMetrics):
            logger.info(
                "[TTS]  ttfb=%.0fms  total=%.0fms  agent_audio=%s  chars=%d",
                m.ttfb * 1000,
                m.duration * 1000,
                _fmt_sec(m.audio_duration),
                m.characters_count,
            )
            if speech_id:
                t = _get(speech_id)
                t.agent_audio_sec += m.audio_duration
                t.tts_ttfb_ms = m.ttfb * 1000
                t.has_tts = True
                _maybe_summarize(speech_id)

    # ── Errors (e.g. LLM timeout, API errors) ─────────────────────────
    @session.on("error")
    def _on_error(ev) -> None:  # type: ignore[no-untyped-def]
        err = ev.error
        source = getattr(ev, "source", "unknown")
        err_type = type(err).__name__
        err_msg = str(err)
        logger.error(
            "\n[ERROR] %s\n  source: %s\n  type: %s\n  message: %s\n%s",
            _THICK,
            source,
            err_type,
            err_msg,
            _THICK,
        )
        if "timeout" in err_msg.lower() or "APITimeoutError" in err_type:
            logger.error(
                "[ERROR] LLM timeout: Groq request took too long. "
                "Check network, Groq status, or increase timeout in the Groq LLM plugin (client param)."
            )


def _extract_text(item: object) -> str:
    content = getattr(item, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(part.text)
        return " ".join(parts)
    return str(content)

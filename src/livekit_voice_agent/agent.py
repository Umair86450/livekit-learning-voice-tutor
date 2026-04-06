from __future__ import annotations

import logging
from typing import Any

from dotenv import load_dotenv
from livekit.agents import AgentSession, JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.stt import StreamAdapter
from livekit.agents.voice import Agent
from livekit.plugins import groq, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel  # too heavy for Railway free tier (OOM kill)

from livekit_voice_agent.config import Settings, get_settings
from livekit_voice_agent.debug import attach_debug_logging
from livekit_voice_agent.health import HealthMonitor
from livekit_voice_agent.log import setup_logging
from livekit_voice_agent.prompts import get_greeting, get_system_prompt
from livekit_voice_agent.rag import RAGEngine
from livekit_voice_agent.stt import LocalWhisperSTT
from livekit_voice_agent.tts import PiperTTS

load_dotenv()

logger = logging.getLogger(__name__)


class VoiceAssistant(Agent):
    def __init__(self, *, settings: Settings, rag_engine: RAGEngine | None = None) -> None:
        super().__init__(
            instructions=get_system_prompt(),
            min_endpointing_delay=settings.min_endpointing_delay,
            max_endpointing_delay=settings.max_endpointing_delay,
            allow_interruptions=settings.allow_interruptions,
        )
        self._settings = settings
        self._rag_engine = rag_engine

    def _needs_explanatory_context(self, query: str) -> bool:
        q = (query or "").lower()
        markers = (
            "explain",
            "why",
            "how",
            "difference",
            "confus",
            "samjha",
            "samjhao",
            "clear",
            "concept",
            "understand",
        )
        return any(m in q for m in markers)

    def _log_tool_call(
        self,
        *,
        tool: str,
        query: str,
        part_slug: str | None,
        chapter_slug: str | None,
        lesson_slug: str | None,
        top_k: int | None = None,
        depth: str | None = None,
    ) -> None:
        if not self._settings.rag_observability_logs:
            return
        logger.info(
            "[RAG_AUDIT] event=tool_call tool=%s top_k=%s depth=%s part=%s chapter=%s lesson=%s query=%r",
            tool,
            top_k if top_k is not None else "-",
            depth if depth is not None else "-",
            part_slug or "-",
            chapter_slug or "-",
            lesson_slug or "-",
            query[:220],
        )

    def _log_tool_result(
        self,
        *,
        tool: str,
        hits: list[Any],
        context_chars: int,
        extra: str | None = None,
    ) -> None:
        if not self._settings.rag_observability_logs:
            return
        top = []
        for h in hits[:5]:
            score = getattr(h, "score", 0.0)
            top.append(f"{h.chunk_id}:{score:.4f}")
        logger.info(
            "[RAG_AUDIT] event=tool_result tool=%s hits=%d context_chars=%d top_chunks=%s extra=%s",
            tool,
            len(hits),
            context_chars,
            ",".join(top) if top else "-",
            extra or "-",
        )

    async def on_enter(self) -> None:
        self.session.say(get_greeting())

    @llm.function_tool(
        name="rag_exact_lookup",
        description="Find exact grounded snippets from knowledge base. Best for specific facts, chapter references, and quoted terms.",
    )
    async def rag_exact_lookup(
        self,
        query: str,
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
    ) -> str:
        q = (query or "").strip()
        p = (part_slug or "").strip() or None
        c = (chapter_slug or "").strip() or None
        lesson = (lesson_slug or "").strip() or None
        self._log_tool_call(
            tool="rag_exact_lookup",
            query=q,
            part_slug=p,
            chapter_slug=c,
            lesson_slug=lesson,
            top_k=self._settings.rag_top_k_exact,
        )
        if not self._settings.rag_enabled:
            return "Knowledge base is disabled."
        if not self._rag_engine or not self._rag_engine.ready:
            return "Knowledge base is not ready yet."

        hits = self._rag_engine.search_exact(
            q,
            part_slug=p,
            chapter_slug=c,
            lesson_slug=lesson,
            top_k=self._settings.rag_top_k_exact,
        )
        if not hits:
            self._log_tool_result(
                tool="rag_exact_lookup",
                hits=[],
                context_chars=0,
                extra="No relevant website content found",
            )
            return "No relevant website content found for this query."

        include_sections = self._needs_explanatory_context(q)
        sections = self._rag_engine.expand_micro_to_sections(hits, max_sections=1) if include_sections else []
        parts: list[str] = []
        total_chars = 0
        for i, hit in enumerate(hits, start=1):
            snippet = hit.text.strip()
            line = f"[{i}] {hit.title} | chunk={hit.chunk_id}\n{snippet}"
            if total_chars + len(line) > self._settings.rag_max_context_chars:
                break
            parts.append(line)
            total_chars += len(line)

        for sec in sections:
            line = f"[section] {sec.title} | chunk={sec.chunk_id}\n{sec.text.strip()}"
            if total_chars + len(line) > self._settings.rag_max_context_chars:
                break
            parts.append(line)
            total_chars += len(line)

        self._log_tool_result(
            tool="rag_exact_lookup",
            hits=hits,
            context_chars=total_chars,
            extra=f"included_sections={len(sections)} include_sections={include_sections}",
        )
        return "\n\n".join(parts) if parts else "No concise context fit the size limit."

    @llm.function_tool(
        name="rag_explain",
        description="Get broader explanatory context from section-level chunks for educational answers.",
    )
    async def rag_explain(
        self,
        query: str,
        depth: str = "standard",
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
    ) -> str:
        q = (query or "").strip()
        p = (part_slug or "").strip() or None
        c = (chapter_slug or "").strip() or None
        lesson = (lesson_slug or "").strip() or None
        if not self._settings.rag_enabled:
            return "Knowledge base is disabled."
        if not self._rag_engine or not self._rag_engine.ready:
            return "Knowledge base is not ready yet."

        depth_key = (depth or "standard").strip().lower()
        top_k_map = {"quick": 1, "standard": self._settings.rag_top_k_explain, "deep": 4}
        top_k = top_k_map.get(depth_key, self._settings.rag_top_k_explain)
        self._log_tool_call(
            tool="rag_explain",
            query=q,
            part_slug=p,
            chapter_slug=c,
            lesson_slug=lesson,
            top_k=top_k,
            depth=depth_key,
        )

        hits = self._rag_engine.search_explain(
            q,
            part_slug=p,
            chapter_slug=c,
            lesson_slug=lesson,
            top_k=top_k,
        )
        if not hits:
            self._log_tool_result(
                tool="rag_explain",
                hits=[],
                context_chars=0,
                extra="No explanatory context found",
            )
            return "No explanatory context found for this query."

        parts: list[str] = []
        total_chars = 0
        for i, hit in enumerate(hits, start=1):
            line = f"[{i}] {hit.title} | chunk={hit.chunk_id}\n{hit.text.strip()}"
            if total_chars + len(line) > self._settings.rag_max_context_chars:
                break
            parts.append(line)
            total_chars += len(line)
        self._log_tool_result(
            tool="rag_explain",
            hits=hits,
            context_chars=total_chars,
        )
        return "\n\n".join(parts) if parts else "No concise context fit the size limit."


def prewarm(proc: JobProcess) -> None:
    settings = get_settings()
    vad_kw: dict[str, float] = {}
    if settings.vad_min_silence_duration is not None:
        vad_kw["min_silence_duration"] = settings.vad_min_silence_duration
    if settings.vad_min_speech_duration is not None:
        vad_kw["min_speech_duration"] = settings.vad_min_speech_duration
    if settings.vad_activation_threshold is not None:
        vad_kw["activation_threshold"] = settings.vad_activation_threshold
    proc.userdata["vad"] = silero.VAD.load(**vad_kw)
    proc.userdata["settings"] = settings
    rag_engine = RAGEngine()
    if settings.rag_enabled:
        try:
            allow_ingest = settings.rag_rebuild_on_start or settings.rag_allow_ingest_on_start
            rag_engine.initialize_from_prepared_data(
                data_dir=settings.rag_data_dir,
                qdrant_url=settings.rag_qdrant_url,
                qdrant_api_key=settings.rag_qdrant_api_key,
                embedding_model_name=settings.rag_embedding_model,
                collection_micro=settings.rag_qdrant_collection_micro,
                collection_section=settings.rag_qdrant_collection_section,
                recreate_collections=settings.rag_qdrant_recreate_collections or settings.rag_rebuild_on_start,
                allow_ingest=allow_ingest,
                batch_size=max(8, settings.rag_batch_size),
            )
            logger.info("RAG initialized from prepared data: %s", settings.rag_data_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG initialization failed: %s", exc)
            if settings.rag_fail_fast_on_init_error:
                raise RuntimeError(f"RAG initialization failed: {exc}") from exc
    proc.userdata["rag_engine"] = rag_engine
    logger.info("Prewarm complete: VAD loaded, settings cached")


def _make_stt(settings: Settings, vad: Any) -> Any:
    if (settings.stt_provider or "groq").lower() == "local":
        local_stt = LocalWhisperSTT(
            model_size=settings.local_stt_model,
            device=settings.local_stt_device,
            compute_type=settings.local_stt_compute_type,
            language=settings.stt_language,   # "en" by default
            download_root=settings.local_stt_download_root,
        )
        return StreamAdapter(stt=local_stt, vad=vad)
    return groq.STT(model=settings.stt_model, language=settings.stt_language)  # "en" by default


async def entrypoint(ctx: JobContext) -> None:
    settings: Settings = ctx.proc.userdata["settings"]
    vad = ctx.proc.userdata["vad"]
    rag_engine: RAGEngine | None = ctx.proc.userdata.get("rag_engine")

    session = AgentSession(
        stt=_make_stt(settings, vad),
        llm=groq.LLM(model=settings.llm_model),
        tts=PiperTTS(model_path=settings.piper_model_path),
        vad=vad,
        # turn_detection=MultilingualModel(),  # disabled: OOM kill on Railway free tier (VAD handles it)
        preemptive_generation=True,
        allow_interruptions=settings.allow_interruptions,
        min_endpointing_delay=settings.min_endpointing_delay,
        max_endpointing_delay=settings.max_endpointing_delay,
        min_interruption_duration=settings.min_interruption_duration,
        min_interruption_words=settings.min_interruption_words,
        false_interruption_timeout=settings.false_interruption_timeout,
        resume_false_interruption=settings.resume_false_interruption,
        discard_audio_if_uninterruptible=settings.discard_audio_if_uninterruptible,
    )

    if settings.debug_metrics_logs:
        attach_debug_logging(session)
    HealthMonitor().attach(session)

    await ctx.connect()
    await session.start(
        agent=VoiceAssistant(settings=settings, rag_engine=rag_engine),
        room=ctx.room,
    )


def main() -> None:
    setup_logging()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )


if __name__ == "__main__":
    main()

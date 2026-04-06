from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    groq_api_key: str = Field(..., description="Groq API key (required for LLM)")

    # STT: "groq" = cloud (needs API key for STT too), "local" = faster-whisper (no API key)
    stt_provider: str = Field(default="groq", description="STT provider: groq | local")
    stt_model: str = "whisper-large-v3-turbo"
    llm_model: str = "llama-3.3-70b-versatile"

    # Language: "en" = English input/output (default)
    stt_language: str = Field(default="en", description="Spoken input language for STT: en | auto")

    # Local STT (faster-whisper) — used when stt_provider=local
    local_stt_model: str = Field(default="base", description="Model size: tiny|base|small|medium|large-v3")
    local_stt_device: str = Field(default="cpu", description="Device: cpu|cuda")
    local_stt_compute_type: str = Field(default="int8", description="Compute: int8|float16|float32")
    local_stt_download_root: str = Field(default="models/whisper", description="Local folder where Whisper model is stored")

    piper_model_path: str = "models/en_US-lessac-high.onnx"

    # Turn detection (EOU): lower = faster response, higher = fewer mid-sentence cuts
    min_endpointing_delay: float = Field(
        default=0.25,
        description="Seconds to wait after speech before considering turn complete (default 0.5 in SDK)",
    )
    max_endpointing_delay: float = Field(
        default=1.5,
        description="Max seconds to wait for user to continue before ending turn (default 3.0 in SDK)",
    )

    # Realistic conversation: when caller speaks while agent is talking, agent stops and listens (barge-in)
    allow_interruptions: bool = Field(
        default=True,
        description="If True, when the caller starts speaking the agent stops and responds to what they said.",
    )

    # AgentSession interruption tuning (realistic feel; SDK defaults if not set)
    min_interruption_duration: float = Field(
        default=0.5,
        description="Min speech length (s) to count as interruption. Lower = more sensitive.",
    )
    min_interruption_words: int = Field(
        default=0,
        description="Min words in transcript to count as interruption. 0 = any speech.",
    )
    false_interruption_timeout: float | None = Field(
        default=2.0,
        description="Seconds to wait for transcript after interruption before resuming (None = disable).",
    )
    resume_false_interruption: bool = Field(
        default=True,
        description="If True, resume agent speech after false_interruption_timeout when no words detected.",
    )
    discard_audio_if_uninterruptible: bool = Field(
        default=True,
        description="Drop user audio while agent is in uninterruptible segment.",
    )

    # Optional VAD (Silero) tuning for faster end-of-utterance; None = use Silero defaults
    vad_min_silence_duration: float | None = Field(
        default=None,
        description="Silence (s) after speech before EOU. Lower = faster reply, e.g. 0.45. None = 0.55.",
    )
    vad_min_speech_duration: float | None = Field(
        default=None,
        description="Min speech (s) to start segment. None = 0.05.",
    )
    vad_activation_threshold: float | None = Field(
        default=None,
        description="Speech probability threshold. None = 0.5; lower = more sensitive.",
    )

    # RAG (prepared docs + Qdrant + open-source embeddings)
    rag_enabled: bool = Field(
        default=False,
        description="Enable RAG tools for grounded responses.",
    )
    rag_data_dir: str = Field(
        default="data/panaversity_rag_prepared",
        description="Prepared JSONL dataset folder (docs/chunks).",
    )
    rag_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Open-source embedding model name (FastEmbed).",
    )
    rag_qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Local Qdrant endpoint URL.",
    )
    rag_qdrant_api_key: str = Field(
        default="",
        description="Optional Qdrant API key.",
    )
    rag_qdrant_collection_micro: str = Field(
        default="panaversity_micro",
        description="Qdrant collection for micro chunks.",
    )
    rag_qdrant_collection_section: str = Field(
        default="panaversity_section",
        description="Qdrant collection for section chunks.",
    )
    rag_rebuild_on_start: bool = Field(
        default=False,
        description="If true, recreate and reingest Qdrant collections from prepared data on startup.",
    )
    rag_allow_ingest_on_start: bool = Field(
        default=False,
        description="If true, allows heavy embedding+ingest during agent startup.",
    )
    rag_fail_fast_on_init_error: bool = Field(
        default=True,
        description="If true, fail worker startup when RAG initialization fails while RAG is enabled.",
    )
    rag_qdrant_recreate_collections: bool = Field(
        default=False,
        description="Drop and recreate Qdrant collections before ingest (typically for offline ingest jobs).",
    )
    rag_batch_size: int = Field(
        default=64,
        description="Batch size for embedding and upsert operations.",
    )
    rag_top_k_exact: int = Field(
        default=3,
        description="Top-k micro chunks for exact lookup tool.",
    )
    rag_top_k_explain: int = Field(
        default=2,
        description="Top-k section chunks for explain tool.",
    )
    rag_max_context_chars: int = Field(
        default=1000,
        description="Maximum context characters returned by tool outputs.",
    )
    rag_observability_logs: bool = Field(
        default=True,
        description="If true, print structured RAG tool-call/result logs.",
    )
    debug_metrics_logs: bool = Field(
        default=False,
        description="If true, enable verbose STT/LLM/TTS latency debug logs.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

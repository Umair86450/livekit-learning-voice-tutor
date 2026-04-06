# RAG and Gotchas

- RAG is optional and disabled by default.
- RAG requires prepared data + running Qdrant.
- `scripts/build_rag_index.py` is deprecated.
- Use prepared-data flow (`scripts/prepare_panaversity_rag_data.py`).
- `livekit-plugins-turn-detector` is intentionally disabled (memory/OOM risk).
- Local STT/TTS requires model artifacts in `models/`.

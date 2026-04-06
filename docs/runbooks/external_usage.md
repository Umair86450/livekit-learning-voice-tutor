# External Usage Checklist

This section consolidates the runtime and deployment knobs mentioned in the README and ensures every integrator knows what to configure when running the agent outside of this workspace.

1. **Groq API key**
   - `GROQ_API_KEY` is required for both STT (`STT_PROVIDER=groq`) and the Groq-hosted LLM model (`llama-3.3-70b-versatile`).
   - Store the key securely (e.g., `.env`, a vault, or GitHub secrets) and never commit it.
   - Restart the agent after acquiring or rotating the key.

2. **LiveKit endpoints**
   - Update `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` for your LiveKit instance.
   - Use `livekit-server --dev` locally or provide the production URL if you're hosting the UI publicly.

3. **STT provider selection**
   - `groq` is the default provider; it trades on low latency and high accuracy but needs internet access and the Groq key.
   - For offline or CPU-only use, set `STT_PROVIDER=local`, specify `LOCAL_STT_MODEL=tiny`, and run `bash scripts/download_models.sh` to fetch Whisper.

4. **TTS setup**
   - Keep `PIPER_MODEL_PATH` pointing to `models/en_US-lessac-high.onnx`.
   - Piper runs locally on CPU; adjust `PIPER_DEVICE` if you add one.

5. **RAG activation**
   - `RAG_ENABLED=true` plus `RAG_DATA_DIR` (default `data/panaversity_rag_prepared`) enables retrieval.
   - Ensure a local Qdrant instance is up (`docker compose -f docker-compose.rag.yml up -d`) before ingestion.
   - Ingest data once per machine with `scripts/rag_ingest_qdrant.py --qdrant-url http://localhost:6333 --data-dir data/panaversity_rag_prepared --recreate`.

6. **Deployment checklist**
   - `uv sync` to install dependencies.
   - Double-check `models/` and `models/whisper/` match your `.env`.
   - Run `tests/test_prompts.py` and `tests/test_agent.py` after any prompt or behavior edits; run `tests/test_config.py` after config changes.

# Environment Contract

## Required
- `GROQ_API_KEY`

## Core Runtime
- `STT_PROVIDER=groq|local`
- `STT_MODEL`, `LLM_MODEL`, `STT_LANGUAGE`
- `PIPER_MODEL_PATH`
- `MIN_ENDPOINTING_DELAY`, `MAX_ENDPOINTING_DELAY`, `ALLOW_INTERRUPTIONS`

## Local Mode
- Ensure whisper model files are present in `models/whisper/`.

## RAG Mode
- Enable `RAG_*` variables.
- Qdrant must be reachable.
- Prepared data directory must exist.

Rule: whenever `Settings` changes, update `.env.example`.

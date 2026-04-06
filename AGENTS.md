# AGENTS.md

## Project
LiveKit-based real-time educational voice assistant: STT -> LLM -> TTS, with optional RAG for grounded answers.

## Response Contract
- Explain in simple beginner-friendly language.
- Use this order in every substantial task:
  1. Problem restatement
  2. Short plan
  3. Changes made
  4. Verification commands + results
  5. Why it works (simple words)
- If context is unclear, state assumptions before editing.

## Non-Negotiables
- Keep changes scoped; avoid unrelated refactors.
- Never commit secrets or keys.
- Prefer config-driven changes over hardcoding.
- Do not use deprecated RAG flow (`scripts/build_rag_index.py`).
- Mandatory documentation sync: whenever behavior, commands, config, workflow, or structure changes, update both `docs/` structure/content and `AGENTS.md` in the same task.
- This documentation sync rule is required and must always be followed.

## High-Signal Files
- `src/livekit_voice_agent/agent.py`
- `src/livekit_voice_agent/config.py`
- `src/livekit_voice_agent/prompts.py`
- `src/livekit_voice_agent/rag.py`
- `tests/`
- `scripts/prepare_panaversity_rag_data.py`

## Core Commands
- Setup: `uv sync`
- Models: `bash scripts/download_models.sh`
- Run (console): `uv run python src/livekit_voice_agent/agent.py console`
- Run (dev): `uv run python src/livekit_voice_agent/agent.py dev`
- Test all: `uv run pytest -v`
- Test focused: `uv run pytest tests/test_<changed_area>.py -v`
- Lint: `uv run ruff check .`
- Deploy start: `python src/livekit_voice_agent/agent.py start`

## Required Checks
- Prompt/behavior changes: run `tests/test_prompts.py` and `tests/test_agent.py`.
- Config/env changes: run `tests/test_config.py` and update `.env.example`.
- RAG changes: run `tests/test_rag.py`.
- Show verification outcomes in final summary.

## Critical Context
- `GROQ_API_KEY` is required.
- STT mode must be explicit: `STT_PROVIDER=groq` or `STT_PROVIDER=local`.
- Local mode requires model files in `models/` and `models/whisper/`.
- RAG requires prepared data + reachable Qdrant.
- `livekit-plugins-turn-detector` stays disabled due to memory/OOM risk.

## Done Criteria
- Requested behavior implemented.
- Relevant tests pass.
- Docs/examples updated for user-visible changes.
- No secrets committed.

## Extended Docs
- See: `docs/architecture/`, `docs/config/`, `docs/runbooks/`, `docs/workflows/`, `docs/standards/`
- Voice tutor behavior rules: `docs/runbooks/voice_tutor_behavior.md`
- External usage checklist: `docs/runbooks/external_usage.md`

## Project Skills
- Project-level skills must live in: `.agents/skills/`
- Current skill: `.agents/skills/rag-production-pipelines/`
- RAG skill must include and maintain: RAG-vs-LLM decision checklist, hallucination warning signs, and Retrieve->Augment->Generate core concept.
- If skill content changes, update skill files and keep this section aligned.

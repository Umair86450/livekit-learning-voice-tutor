# Verification Matrix

- Prompt/behavior change -> `tests/test_prompts.py`, `tests/test_agent.py`
- Config/env change -> `tests/test_config.py` + `.env.example` sync
- RAG change -> `tests/test_rag.py`
- Broad logic change -> targeted pytest + `uv run pytest -v`
- Style-sensitive edits -> `uv run ruff check .`

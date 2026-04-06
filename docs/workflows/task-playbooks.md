# Task Playbooks

## Tests
- Add/update focused tests in `tests/` for changed behavior.
- Run targeted tests, then full suite if feasible.

## Behavior Changes
- Touch `prompts.py` and/or `agent.py`.
- Validate prompt/agent tests.

## Config Changes
- Update `config.py` + `.env.example` + docs.
- Run `tests/test_config.py`.

## RAG Changes
- Update retrieval logic/tool shape.
- Run `tests/test_rag.py`.

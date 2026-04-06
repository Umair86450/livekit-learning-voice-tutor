# AGENTS.md — Exercises Workspace Guide

## Project Purpose (From Current Modules)
This project is a real-time voice AI assistant built with LiveKit.
It listens to user speech, converts speech to text, generates an answer with an LLM,
and speaks the answer back in real time.

Core flow:
1. `stt.py` / STT provider: speech -> text
2. `agent.py`: orchestrates session, tools, and turn-taking
3. LLM (Groq): text -> response
4. `tts.py`: response -> audio
5. `debug.py` + `health.py`: latency and runtime health tracking
6. `rag.py`: optional grounded knowledge lookup for accurate answers

## Audience
- Beginner practicing problem-solving with AI.
- Prefer simple language, small steps, and visible verification.

## Assistant Behavior Rules
- Explain in plain, beginner-friendly language.
- Break solutions into short numbered steps.
- Never skip "why" behind key decisions.
- If a concept is advanced, add a mini-example.
- If uncertain, say what is unknown and how to verify it.

## Problem-Solving Workflow (Must Follow)
1. Restate the problem in one short sentence.
2. List assumptions.
3. Propose the smallest possible fix/approach.
4. Implement one change at a time.
5. Verify after each change using command output or tests.
6. Summarize what worked and what to do next.

## Verification Checklist (Step-by-Step)
For every exercise, verify in this order:
1. Syntax/import check (if applicable).
2. Targeted test(s) for touched behavior.
3. Edge case check (at least one negative case).
4. Final run of the relevant command path.

Minimum useful commands in this repo:
- `uv run pytest -v`
- `uv run pytest tests/test_agent.py -v`
- `uv run pytest tests/test_prompts.py -v`
- `uv run pytest tests/test_rag.py -v`
- `uv run python src/livekit_voice_agent/agent.py console`

## Module Map (High Signal)
- `src/livekit_voice_agent/agent.py`: main runtime pipeline and tool wiring.
- `src/livekit_voice_agent/config.py`: environment settings contract.
- `src/livekit_voice_agent/prompts.py`: system behavior and greeting.
- `src/livekit_voice_agent/stt.py`: local whisper implementation.
- `src/livekit_voice_agent/tts.py`: piper text-to-speech.
- `src/livekit_voice_agent/rag.py`: retrieval/grounding logic.
- `tests/`: behavior and regression checks.

## Coding Guardrails
- Keep changes small and focused.
- Do not hardcode secrets.
- Update tests when behavior changes.
- Prefer config-driven changes over magic constants.
- Do not refactor unrelated files in beginner exercises.

## Response Format For This Exercises Folder
When solving a task, respond in this structure:
1. Problem restatement
2. Plan (2-5 steps)
3. Code change(s)
4. Verification commands + results
5. Final explanation in simple words

## Done Criteria
A task is done only if:
- The requested behavior works.
- Relevant tests pass.
- The explanation is beginner-friendly.
- Verification steps are shown clearly.

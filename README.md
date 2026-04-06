# LiveKit Voice Agent

A real-time AI **voice educational assistant**. Students can ask questions in English, explain their confusion in real time, and get clear step-by-step voice guidance.

**Current language:** English
**Current STT:** Local Whisper (`tiny` model, CPU)
**Current TTS:** Piper (`en_US-lessac-high`, local CPU)
**Current LLM:** Groq `llama-3.3-70b-versatile`

## Architecture overview

| Layer | Purpose | Notes |
|-------|---------|-------|
| **LiveKit** (web RTC) | Hosts the browser/agent call with mic + speaker | Use `uv run python src/livekit_voice_agent/agent.py dev` against a LiveKit server for UI playback. |
| **STT** | Converts spoken English into text | Default is Groq Whisper (`STT_PROVIDER=groq`); local Whisper is available for offline/dev use (`STT_PROVIDER=local`, `LOCAL_STT_MODEL=tiny`). |
| **LLM** | Groq `llama-3.3-70b-versatile` on Groq Cloud | Requires `GROQ_API_KEY`, enforces the tutor prompt, and powers the `rag_exact_lookup`/`rag_explain` flows. |
| **TTS** | Piper local engine (`en_US-lessac-high`) | Reads answers back through your speakers; keep `PIPER_MODEL_PATH` pointing to `models/en_US-lessac-high.onnx`. |
| **RAG (optional)** | Grounded retrieval with Qdrant + embeddings | Toggle via `.env`: `RAG_ENABLED=true`, point to `data/panaversity_rag_prepared`, run `scripts/rag_ingest_qdrant.py`, and ensure a Qdrant instance is reachable. |

## External usage guidance

1. **Groq API key** – `GROQ_API_KEY` is mandatory for STT/LLM. Store it in `.env` (never commit the key) and reload the agent whenever it changes.
2. **LiveKit credentials** – Use `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` from LiveKit Cloud or your local server. Update `.env` to match the deployment and restart.
3. **STT provider selection** – `groq` is the production default (fast, accurate) and needs internet access. If you must run offline, switch to `local` and download Whisper with `bash scripts/download_models.sh`; `LOCAL_STT_MODEL=tiny` is the real-time option on CPU.
4. **RAG prerequisites** – Prepare `data/panaversity_rag_prepared/` (JSONL chunks + manifest). Start Qdrant (`docker compose -f docker-compose.rag.yml up -d`) before ingestion. Follow `scripts/rag_ingest_qdrant.py` to populate vectors, then set `RAG_QDRANT_URL` in `.env`.
5. **Deployment checklist** – Before pushing to production:
   - Run `uv sync` to install deps.
   - Ensure `models/` and `models/whisper/` match `.env`.
   - Verify `tests/test_prompts.py` and `tests/test_agent.py` pass after any prompt or behavior edits.

## Quick Start

## Agent behavior

The embedded system prompt frames the agent as a calm, encouraging **AI voice tutor** that keeps replies speech-friendly and concise. It always speaks in natural English, avoids markdown, and reacts immediately to new user intent. For factual or chapter-based questions it calls `rag_exact_lookup`, and for conceptual/deeper understanding it prioritizes `rag_explain` (using the `depth` parameter when the learner requests quick or deep answers). When possible it chains both: an exact lookup first, then an explain step. The agent never invents facts—if RAG output is weak it explains that evidence is insufficient and asks for a focused clarification. Practice requests trigger short examples or mini-exercises, and grounded answers mention the source title and chunk id briefly in spoken form before wrapping up politely.

---

## Quick Start

```bash
uv run python src/livekit_voice_agent/agent.py console
```

This opens a direct mic + speaker session. Speak in English and the agent replies in English.

**Client demo (browser + same network ya remote):** See **[DEMO_SETUP.md](DEMO_SETUP.md)** for minimal steps to go live and let clients test.

---

## Full Setup Guide

### Step 1 — Download Models

```bash
bash scripts/download_models.sh
```

Downloads:
- Piper TTS voice models → `models/`
- Local Whisper STT model → `models/whisper/`

### Step 2 — Configure `.env`

Copy the example and fill in your Groq API key:
```bash
cp .env.example .env
```

Set your key:
```env
GROQ_API_KEY=your_groq_key_here
```

### Step 3 — Run

**Console mode** (mic + speaker, no browser):
```bash
uv run python src/livekit_voice_agent/agent.py console
```

**Dev mode** (connects to LiveKit server, use with browser playground):
```bash
# Terminal 1 — start LiveKit server
livekit-server --dev --bind 0.0.0.0

# Terminal 2 — start agent
uv run python src/livekit_voice_agent/agent.py dev
```

Then open `https://agents-playground.livekit.io` and connect with:
- URL: `ws://localhost:7880`
- API Key: `devkey`
- API Secret: `secret`

---

## STT Configuration — Groq vs Local Whisper

Open `.env`. Only **one STT option** should be active at a time.

### Option A — Local Whisper (currently active)

```env
STT_PROVIDER=local
LOCAL_STT_MODEL=tiny
LOCAL_STT_DEVICE=cpu
LOCAL_STT_COMPUTE_TYPE=int8
LOCAL_STT_DOWNLOAD_ROOT=models/whisper
```

**Model size options and CPU benchmark results:**

| Model    | Per-Turn Latency | Load Time | Accuracy       | Use When                        |
|----------|-----------------|-----------|----------------|---------------------------------|
| `tiny`   | ~462ms ✓        | 11s       | Basic          | CPU only option that is usable  |
| `base`   | ~4000ms ✗       | 1.25s     | Better         | Too slow on CPU                 |
| `medium` | ~5000ms ✗       | 207s      | Best (local)   | Needs GPU (CUDA)                |
| `large-v3`| ~8000ms ✗      | 300s+     | Best (local)   | Needs GPU (CUDA)                |

> **Important:** On a CPU-only Mac, only `tiny` is fast enough for real-time voice. `base` and above add 4–5 seconds of delay per response, which breaks the conversation flow.

**To change the model size**, edit one line in `.env`:
```env
LOCAL_STT_MODEL=tiny    # fast, less accurate
LOCAL_STT_MODEL=base    # slow on CPU, needs GPU
```

**To download the selected model**, run:
```bash
bash scripts/download_models.sh
```
The script reads `LOCAL_STT_MODEL` from `.env` and downloads that model to `models/whisper/`. If the model already exists, it skips the download.

### Option B — Groq STT (cloud, recommended for production)

To switch to Groq STT: comment out Option A and uncomment Option B in `.env`:

```env
# OPTION A: comment these out
# STT_PROVIDER=local
# LOCAL_STT_MODEL=tiny
# LOCAL_STT_DEVICE=cpu
# LOCAL_STT_COMPUTE_TYPE=int8
# LOCAL_STT_DOWNLOAD_ROOT=models/whisper

# OPTION B: uncomment these
STT_PROVIDER=groq
STT_MODEL=whisper-large-v3-turbo
```

| Feature          | Local Whisper (tiny) | Groq Whisper (cloud)         |
|------------------|---------------------|------------------------------|
| Latency          | ~462ms              | ~50–100ms                    |
| Accuracy         | Basic               | Excellent                    |
| English support  | Good                | Very good                    |
| Internet needed  | No                  | Yes                          |
| API cost         | Free                | Groq free tier (with limits) |

---

## Language Configuration

```env
STT_LANGUAGE=en   # English input/output
```

Also update `PIPER_MODEL_PATH` to match:

```env
PIPER_MODEL_PATH=models/en_US-lessac-high.onnx
```

Restart the agent after any language change.

---

## RAG Setup (Production-Ready, Shareable Repo Workflow)

Use this workflow when you share the repo with others:
- Share code + prepared JSONL data.
- Do not share prebuilt vector DB files.
- Each developer builds vectors on their own machine once.

### What to include in the repo

```text
data/panaversity_rag_prepared/
  docs.jsonl
  chunks_section.jsonl
  chunks_micro.jsonl
```

### Receiver-side prerequisites

1. Docker Desktop installed and running.
2. Python/uv installed.
3. `GROQ_API_KEY` set in `.env`.

### Receiver-side setup (run once per machine)

1) Install Python deps:

```bash
uv sync
```

2) Start persistent Qdrant:

```bash
docker compose -f docker-compose.rag.yml up -d
```

3) Build vectors in Qdrant (one-time ingest):

```bash
uv run python scripts/rag_ingest_qdrant.py \
  --data-dir data/panaversity_rag_prepared \
  --qdrant-url http://localhost:6333 \
  --recreate \
  --batch-size 128
```

4) Verify counts:

```bash
curl -s http://localhost:6333/collections/panaversity_micro | jq '.result.points_count'
curl -s http://localhost:6333/collections/panaversity_section | jq '.result.points_count'
```

Expected:
- `panaversity_micro`: `18194`
- `panaversity_section`: `5099`

### Runtime `.env` (production-safe)

```env
RAG_ENABLED=true
RAG_DATA_DIR=data/panaversity_rag_prepared
RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAG_QDRANT_URL=http://localhost:6333
RAG_QDRANT_RECREATE_COLLECTIONS=false
RAG_ALLOW_INGEST_ON_START=false
RAG_TOP_K_EXACT=5
RAG_TOP_K_EXPLAIN=3
RAG_MAX_CONTEXT_CHARS=1500
```

Why these flags:
- `RAG_QDRANT_RECREATE_COLLECTIONS=false` prevents accidental data wipe on startup.
- `RAG_ALLOW_INGEST_ON_START=false` prevents heavy prewarm ingest and startup timeouts.

### Start agent

```bash
uv run python src/livekit_voice_agent/agent.py console
```

Tools exposed to the LLM:
- `rag_exact_lookup` for precise fact lookup from micro chunks.
- `rag_explain` for broader educational context from section chunks.

### If data changes later

If you regenerate `data/panaversity_rag_prepared`, re-run one-time ingest:

```bash
uv run python scripts/rag_ingest_qdrant.py \
  --data-dir data/panaversity_rag_prepared \
  --qdrant-url http://localhost:6333 \
  --recreate
```

Then run agent normally.

---

## Realistic Voice Parameters — Full Explanation

These parameters control how natural and human-like the conversation feels. They are set in `.env`.

---

### EOU — End of Utterance (When does the agent decide the user has finished speaking?)

EOU is the moment the agent decides "the user has stopped talking — now I should reply." Getting this right is the most important factor for natural conversation feel.

#### `MIN_ENDPOINTING_DELAY`
```env
MIN_ENDPOINTING_DELAY=0.25
```
**What it does:** Minimum time (in seconds) the agent waits after the user goes silent before treating the turn as complete and generating a reply.

| Value  | Effect                                                              |
|--------|---------------------------------------------------------------------|
| `0.1`  | Very fast reply — may cut user off mid-sentence (too aggressive)    |
| `0.25` | Balanced — default, works well for most conversations              |
| `0.35` | Slightly more patient — good if users often get cut off            |
| `0.5`  | Conservative — SDK default, feels slower                           |

**Tune this when:**
- Agent replies too quickly and cuts the user off → increase to `0.35`
- Agent feels slow to respond → decrease to `0.2`

---

#### `MAX_ENDPOINTING_DELAY`
```env
MAX_ENDPOINTING_DELAY=1.5
```
**What it does:** Maximum time the agent will wait for the user to continue speaking after a pause. Even if the user pauses mid-sentence, the agent will not wait longer than this value.

| Value  | Effect                                                              |
|--------|---------------------------------------------------------------------|
| `1.0`  | Aggressive — does not give much room for natural pauses            |
| `1.5`  | Balanced — default                                                 |
| `2.0`  | Patient — good for users who pause to think                        |
| `3.0`  | SDK default — often feels too slow for a phone call                |

**Tune this when:**
- Agent cuts the user off when they pause to think → increase to `2.0`
- Responses feel slow because the agent always waits too long → decrease to `1.2`

---

### Barge-in / Interruptions (Can the caller interrupt the agent?)

#### `ALLOW_INTERRUPTIONS`
```env
ALLOW_INTERRUPTIONS=true
```
**What it does:** When `true`, if the caller speaks while the agent is talking, the agent stops immediately and responds to what the caller just said. This is called **barge-in** and is essential for natural phone conversations.

| Value   | Effect                                                              |
|---------|---------------------------------------------------------------------|
| `true`  | Realistic — agent stops and listens when caller speaks (recommended)|
| `false` | Agent finishes its full response before listening again            |

---

### Optional Interruption Tuning

These are commented out in `.env` by default. The defaults work well — only uncomment and adjust if you have a specific problem.

```env
# MIN_INTERRUPTION_DURATION=0.5
# MIN_INTERRUPTION_WORDS=0
# FALSE_INTERRUPTION_TIMEOUT=2.0
# RESUME_FALSE_INTERRUPTION=true
# DISCARD_AUDIO_IF_UNINTERRUPTIBLE=true
```

#### `MIN_INTERRUPTION_DURATION`
```env
# MIN_INTERRUPTION_DURATION=0.5
```
Minimum length of speech (in seconds) that counts as a real interruption. Background noise shorter than this is ignored.

| Value  | Effect                                                              |
|--------|---------------------------------------------------------------------|
| `0.2`  | Very sensitive — any brief sound triggers interruption             |
| `0.5`  | Default — ignores short noises under 0.5 seconds                  |
| `0.8`  | Less sensitive — only clear speech stops the agent                |

**Uncomment and lower** if the agent is not stopping when the caller speaks.
**Uncomment and raise** if background noise or brief sounds are incorrectly stopping the agent.

---

#### `MIN_INTERRUPTION_WORDS`
```env
# MIN_INTERRUPTION_WORDS=0
```
Minimum number of transcribed words required to count as an interruption. `0` means any detected speech counts.

| Value  | Effect                                                              |
|--------|---------------------------------------------------------------------|
| `0`    | Any speech triggers interruption (default)                         |
| `1`    | At least one word must be recognized                               |
| `2`    | At least two words required — reduces false interruptions          |

**Uncomment and set to `1` or `2`** if the agent is stopping due to background noise that gets incorrectly transcribed as a word.

---

#### `FALSE_INTERRUPTION_TIMEOUT`
```env
# FALSE_INTERRUPTION_TIMEOUT=2.0
```
When the agent detects an interruption and stops speaking, it waits this many seconds for a transcript from STT. If no words come through within this time, it treats it as a **false interruption** (background noise).

| Value   | Effect                                                             |
|---------|--------------------------------------------------------------------|
| `1.0`   | Quick recovery — agent resumes fast after false interruption       |
| `2.0`   | Default — 2 second wait before deciding it was a false interrupt  |
| `null`  | Disable — agent never resumes after stopping (not recommended)    |

---

#### `RESUME_FALSE_INTERRUPTION`
```env
# RESUME_FALSE_INTERRUPTION=true
```
**What it does:** If a false interruption is detected (no words came through within `FALSE_INTERRUPTION_TIMEOUT`), the agent resumes where it left off.

| Value   | Effect                                                             |
|---------|--------------------------------------------------------------------|
| `true`  | Agent resumes speech after false interruption (recommended)        |
| `false` | Agent stays silent after any interruption, even false ones        |

---

#### `DISCARD_AUDIO_IF_UNINTERRUPTIBLE`
```env
# DISCARD_AUDIO_IF_UNINTERRUPTIBLE=true
```
During certain moments (e.g. while the agent is still generating the LLM response), the agent is in an "uninterruptible" state. This setting controls what happens to user audio captured during that window.

| Value   | Effect                                                             |
|---------|--------------------------------------------------------------------|
| `true`  | User audio during uninterruptible window is dropped (recommended) |
| `false` | User audio is buffered and processed after — can feel laggy       |

---

### Optional VAD Tuning

VAD (Voice Activity Detection) runs on Silero and detects when the user is actively speaking vs silent. These are commented out — leave them unset unless you have audio quality issues.

```env
# VAD_MIN_SILENCE_DURATION=
# VAD_MIN_SPEECH_DURATION=
# VAD_ACTIVATION_THRESHOLD=
```

#### `VAD_MIN_SILENCE_DURATION`
How long (in seconds) of silence after speech before the VAD marks the user as done speaking.

| Value    | Effect                                                             |
|----------|--------------------------------------------------------------------|
| unset    | Silero default: 0.55s                                             |
| `0.35`   | Faster EOU — reply starts sooner                                  |
| `0.45`   | Slightly faster than default                                      |
| `0.7`    | More patient — good for users who pause between phrases           |

**Uncomment and lower** (e.g. `0.45`) if you want faster response times.
**Uncomment and raise** (e.g. `0.7`) if the agent is cutting off users who speak with natural pauses.

---

#### `VAD_MIN_SPEECH_DURATION`
Minimum speech length (in seconds) before VAD considers it a real speech segment (not noise).

| Value    | Effect                                                             |
|----------|--------------------------------------------------------------------|
| unset    | Silero default: 0.05s                                             |
| `0.1`    | Slightly less sensitive to very short sounds                      |
| `0.2`    | Ignores brief sounds and coughs                                   |

---

#### `VAD_ACTIVATION_THRESHOLD`
Probability score (0.0–1.0) that Silero requires to classify audio as speech.

| Value    | Effect                                                             |
|----------|--------------------------------------------------------------------|
| unset    | Silero default: 0.5                                               |
| `0.3`    | More sensitive — picks up quiet speech but also more noise        |
| `0.6`    | Less sensitive — requires clearer speech signal                   |
| `0.8`    | Strict — good for clean audio environments only                   |

---

## Recommended Settings by Scenario

| Scenario | Recommended Changes |
|----------|---------------------|
| Feeling laggy / slow replies | `MIN_ENDPOINTING_DELAY=0.2`, `VAD_MIN_SILENCE_DURATION=0.4` |
| Agent cuts caller off | `MIN_ENDPOINTING_DELAY=0.35`, `MAX_ENDPOINTING_DELAY=2.0` |
| Background noise causing false stops | `MIN_INTERRUPTION_DURATION=0.8`, `MIN_INTERRUPTION_WORDS=1` |
| Noisy environment | `VAD_ACTIVATION_THRESHOLD=0.6` |
| Quiet mic / faint voice | `VAD_ACTIVATION_THRESHOLD=0.3` |

---

## Run Tests

```bash
uv run pytest tests/ -v
```

Tests cover: agent setup, prompts, TTS chunking, STT, debug latency tracking, health monitoring.

---

## Project Structure

```
Livekit-Voice-Agent/
├── src/livekit_voice_agent/
│   ├── agent.py        # main agent + LiveKit worker
│   ├── config.py       # all settings loaded from .env
│   ├── prompts.py      # system prompt + greeting (English)
│   ├── tts.py          # Piper TTS (local CPU, chunked streaming)
│   ├── stt.py          # local Whisper STT (faster-whisper)
│   ├── debug.py        # per-turn latency breakdown logging
│   ├── health.py       # uptime, error count, latency monitoring
│   └── log.py          # logging setup
├── tests/              # pytest test suite
├── scripts/
│   ├── download_models.sh   # downloads Piper + Whisper models
│   └── run_local.sh
├── models/
│   ├── en_US-lessac-high.onnx     # English TTS voice
│   └── whisper/                   # local Whisper STT models
├── .env                # configuration (git ignored)
└── .env.example        # template
```

---

## Requirements

| Requirement       | Status                              |
|-------------------|-------------------------------------|
| Groq API Key      | Required — set in `.env`            |
| Piper TTS model   | Download via `scripts/download_models.sh` |
| Whisper model     | Download via `scripts/download_models.sh` |
| LiveKit server    | Only needed for `dev` mode          |
| Python 3.12+      | Managed via `uv`                    |





## Summary Table

| Step | Command | Terminal |
|------|---------|---------|
| 1. Server start | `livekit-server --dev --bind 0.0.0.0` | Terminal 1 |
| 2. Agent start | `uv run python src/livekit_voice_agent/agent.py dev` | Terminal 2 |
| 3. Token generate | `livekit-cli token create --api-key devkey --api-secret secret --room my-room --identity user1 --join` | Terminal 3 |
| 4. Browser | https://agents-playground.livekit.io > Manual tab | Browser |

URL:   wss://test-voice-shu02yqv.livekit.cloud

       wss://test-voice-shu02yqv.livekit.cloud

 livekit-cli token create \
    --api-key APIxowg59cUZEki \
    --api-secret N4xTWvyYkvkfifZtefz6yPC3Brr2b5ZGXcbGOthnP90A \
    --room demo-room \
    --identity client1 \
    --join \
    --valid-for 2m




     livekit-cli token create \
       --api-key APIwu93FqKm3NFu \
       --api-secret rAGlD3AgSJhOE8uQSDc9Aek30vKMzpF171a0jaNyOaZ \
       --room demo-room \
       --identity client1 \
       --join \
       --valid-for 5m

livekit-cli token create \
         --api-key APIwu93FqKm3NFu \
         --api-secret rAGlD3AgSJhOE8uQSDc9Aek30vKMzpF171a0jaNyOaZ \
         --room demo-room \
         --identity client1 \
        --join \
        --valid-for 10m



     livekit-cli room list \
        --url wss://test-voice-shu02yqv.livekit.cloud \
        --api-key APIwu93FqKm3NFu \
        --api-secret rAGlD3AgSJhOE8uQSDc9Aek30vKMzpF171a0jaNyOaZ

 livekit-cli worker list \
        --url wss://test-voice-shu02yqv.livekit.cloud \
        --api-key APIwu93FqKm3NFu \
        --api-secret rAGlD3AgSJhOE8uQSDc9Aek30vKMzpF171a0jaNyOaZ

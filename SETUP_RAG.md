# RAG Quick Setup (Copy-Paste)

This guide is for a fresh machine after cloning the repo.

## 1) Install dependencies

```bash
uv sync
```

## 2) Start Qdrant (persistent)

```bash
docker compose -f docker-compose.rag.yml up -d
```

## 3) One-time vector ingest

```bash
uv run python scripts/rag_ingest_qdrant.py \
  --data-dir data/panaversity_rag_prepared \
  --qdrant-url http://localhost:6333 \
  --recreate \
  --batch-size 128
```

## 4) Verify ingest counts

```bash
curl -s http://localhost:6333/collections/panaversity_micro | jq '.result.points_count'
curl -s http://localhost:6333/collections/panaversity_section | jq '.result.points_count'
```

Expected:
- `18194` for `panaversity_micro`
- `5099` for `panaversity_section`

## 5) Configure `.env`

Set these values:

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

## 6) Run agent

```bash
uv run python src/livekit_voice_agent/agent.py console
```

## Re-ingest (only when data changes)

If `data/panaversity_rag_prepared` changes:

```bash
uv run python scripts/rag_ingest_qdrant.py \
  --data-dir data/panaversity_rag_prepared \
  --qdrant-url http://localhost:6333 \
  --recreate
```


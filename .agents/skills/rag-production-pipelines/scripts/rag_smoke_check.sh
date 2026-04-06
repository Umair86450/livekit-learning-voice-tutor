#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
RAG_DATA_DIR="${RAG_DATA_DIR:-data/panaversity_rag_prepared}"

echo "[check] Qdrant endpoint: ${QDRANT_URL}"
curl -fsS "${QDRANT_URL}/collections" >/dev/null
echo "[ok] Qdrant reachable"

echo "[check] Prepared data directory: ${RAG_DATA_DIR}"
if [ ! -d "${RAG_DATA_DIR}" ]; then
  echo "[fail] Missing directory: ${RAG_DATA_DIR}" >&2
  exit 1
fi

for f in docs.jsonl chunks_section.jsonl chunks_micro.jsonl; do
  if [ ! -f "${RAG_DATA_DIR}/${f}" ]; then
    echo "[fail] Missing file: ${RAG_DATA_DIR}/${f}" >&2
    exit 1
  fi
done

echo "[ok] Required prepared data files present"
echo "[done] RAG smoke checks passed"

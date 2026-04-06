#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit_voice_agent.rag import RAGEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline ingest prepared RAG JSONL data into Qdrant.")
    parser.add_argument("--data-dir", default="data/panaversity_rag_prepared")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", default="")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--collection-micro", default="panaversity_micro")
    parser.add_argument("--collection-section", default="panaversity_section")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collections before ingest.",
    )
    args = parser.parse_args()

    engine = RAGEngine()
    engine.initialize_from_prepared_data(
        data_dir=args.data_dir,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key or None,
        embedding_model_name=args.embedding_model,
        collection_micro=args.collection_micro,
        collection_section=args.collection_section,
        recreate_collections=args.recreate,
        allow_ingest=True,
        batch_size=max(8, args.batch_size),
    )
    print("RAG ingestion complete. Ready:", engine.ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

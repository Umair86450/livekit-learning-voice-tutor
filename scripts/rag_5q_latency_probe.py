#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from livekit_voice_agent.rag import RAGEngine, RAGHit


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_hits(hits: list[RAGHit], *, top_n: int = 5) -> list[str]:
    out: list[str] = []
    for h in hits[:top_n]:
        out.append(f"{h.chunk_id}:{h.score:.4f}")
    return out


def _preview(text: str, *, max_chars: int = 220) -> str:
    t = " ".join((text or "").strip().split())
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def _build_exact_response(hits: list[RAGHit], *, max_chars: int = 1500) -> str:
    if not hits:
        return "No relevant website content found for this query."
    parts: list[str] = []
    total = 0
    for i, h in enumerate(hits, start=1):
        line = f"[{i}] {h.title} | chunk={h.chunk_id}\n{h.text.strip()}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n\n".join(parts) if parts else "No concise context fit the size limit."


def _build_explain_response(hits: list[RAGHit], *, max_chars: int = 1500) -> str:
    if not hits:
        return "No explanatory context found for this query."
    parts: list[str] = []
    total = 0
    for i, h in enumerate(hits, start=1):
        line = f"[{i}] {h.title} | chunk={h.chunk_id}\n{h.text.strip()}"
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)
    return "\n\n".join(parts) if parts else "No concise context fit the size limit."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask 1 question and log RAG retrieval latency + tool responses.",
    )
    parser.add_argument("--data-dir", default="data/panaversity_rag_prepared")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", default="")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--collection-micro", default="panaversity_micro")
    parser.add_argument("--collection-section", default="panaversity_section")
    parser.add_argument("--top-k-exact", type=int, default=5)
    parser.add_argument("--top-k-explain", type=int, default=5)
    parser.add_argument("--part-slug", default="")
    parser.add_argument("--chapter-slug", default="")
    parser.add_argument("--lesson-slug", default="")
    parser.add_argument("--question", default="", help="Single question; if empty script asks interactively.")
    parser.add_argument("--max-context-chars", type=int, default=1500)
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    engine = RAGEngine()
    engine.initialize_from_prepared_data(
        data_dir=args.data_dir,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key or None,
        embedding_model_name=args.embedding_model,
        collection_micro=args.collection_micro,
        collection_section=args.collection_section,
        recreate_collections=False,
        allow_ingest=False,
        batch_size=64,
    )
    if not engine.ready:
        print("RAG engine not ready.")
        return 1

    part_slug = args.part_slug.strip() or None
    chapter_slug = args.chapter_slug.strip() or None
    lesson_slug = args.lesson_slug.strip() or None

    question = (args.question or "").strip()
    while not question:
        question = input("Q> ").strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / f"rag_latency_probe_single_{ts}.jsonl"
    csv_path = log_dir / f"rag_latency_probe_single_{ts}.csv"

    turn_started = time.perf_counter()
    t0 = time.perf_counter()
    exact_hits = engine.search_exact(
        question,
        part_slug=part_slug,
        chapter_slug=chapter_slug,
        lesson_slug=lesson_slug,
        top_k=max(1, args.top_k_exact),
    )
    exact_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    explain_hits = engine.search_explain(
        question,
        part_slug=part_slug,
        chapter_slug=chapter_slug,
        lesson_slug=lesson_slug,
        top_k=max(1, args.top_k_explain),
    )
    explain_ms = (time.perf_counter() - t1) * 1000.0

    exact_response = _build_exact_response(exact_hits, max_chars=args.max_context_chars)
    explain_response = _build_explain_response(explain_hits, max_chars=args.max_context_chars)
    agent_response = exact_response if exact_hits else explain_response
    total_ms = (time.perf_counter() - turn_started) * 1000.0

    row = {
        "timestamp_utc": _now_iso(),
        "q_index": 1,
        "question": question,
        "filters": {
            "part_slug": part_slug,
            "chapter_slug": chapter_slug,
            "lesson_slug": lesson_slug,
        },
        "latency_ms": {
            "exact_lookup_ms": round(exact_ms, 2),
            "explain_lookup_ms": round(explain_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        "counts": {
            "exact_hits": len(exact_hits),
            "explain_hits": len(explain_hits),
        },
        "top_chunks": {
            "exact": _fmt_hits(exact_hits, top_n=5),
            "explain": _fmt_hits(explain_hits, top_n=5),
        },
        "agent_response": agent_response,
        "exact_response": exact_response,
        "explain_response": explain_response,
    }

    print(f"exact={exact_ms:.1f}ms explain={explain_ms:.1f}ms total={total_ms:.1f}ms")
    print(f"exact_hits={len(exact_hits)} explain_hits={len(explain_hits)}")

    print("\nRetrieved (exact, top-5):")
    for i, h in enumerate(exact_hits[:5], start=1):
        print(f"[{i}] {h.chunk_id} score={h.score:.4f} title={h.title}")
        print(f"    {_preview(h.text)}")

    print("\nRetrieved (section/explain, top-5):")
    for i, h in enumerate(explain_hits[:5], start=1):
        print(f"[{i}] {h.chunk_id} score={h.score:.4f} title={h.title}")
        print(f"    {_preview(h.text)}")

    print("\nAgent response (tool output):")
    print(agent_response[:1200] + ("..." if len(agent_response) > 1200 else ""))

    with jsonl_path.open("w", encoding="utf-8") as jf:
        jf.write(json.dumps(row, ensure_ascii=True))
        jf.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "timestamp_utc",
                "q_index",
                "question",
                "exact_lookup_ms",
                "explain_lookup_ms",
                "total_ms",
                "exact_hits",
                "explain_hits",
                "top_exact",
                "top_explain",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": row["timestamp_utc"],
                "q_index": row["q_index"],
                "question": row["question"],
                "exact_lookup_ms": row["latency_ms"]["exact_lookup_ms"],
                "explain_lookup_ms": row["latency_ms"]["explain_lookup_ms"],
                "total_ms": row["latency_ms"]["total_ms"],
                "exact_hits": row["counts"]["exact_hits"],
                "explain_hits": row["counts"]["explain_hits"],
                "top_exact": " | ".join(row["top_chunks"]["exact"]),
                "top_explain": " | ".join(row["top_chunks"]["explain"]),
            }
        )

    print("\nLatency summary:")
    print(f"- exact lookup : {row['latency_ms']['exact_lookup_ms']:.2f} ms")
    print(f"- explain lookup: {row['latency_ms']['explain_lookup_ms']:.2f} ms")
    print(f"- total turn   : {row['latency_ms']['total_ms']:.2f} ms")
    print(f"\nLogs written:\n- {jsonl_path}\n- {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

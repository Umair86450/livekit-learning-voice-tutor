#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit_voice_agent.rag import RAGEngine, RAGHit


TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def _coverage_score(query: str, text: str) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0
    t = _tokenize(text)
    return len(q.intersection(t)) / len(q)


def _parse_input(raw: str) -> tuple[str, list[str]]:
    # Format:
    #   question only
    #   question || keyword1, keyword2, keyword3
    if "||" not in raw:
        return raw.strip(), []
    question, expected = raw.split("||", 1)
    expected_terms = [x.strip().lower() for x in expected.split(",") if x.strip()]
    return question.strip(), expected_terms


def _expected_match_ratio(expected_terms: list[str], text: str) -> float:
    if not expected_terms:
        return 0.0
    content = text.lower()
    matches = sum(1 for t in expected_terms if t in content)
    return matches / len(expected_terms)


def _format_hit(hit: RAGHit, *, idx: int, query: str, expected_terms: list[str]) -> str:
    preview = " ".join(hit.text.strip().split())
    if len(preview) > 260:
        preview = preview[:260].rstrip() + "..."
    cov = _coverage_score(query, hit.text)
    exp = _expected_match_ratio(expected_terms, hit.text) if expected_terms else 0.0
    return (
        f"[{idx}] score={hit.score:.4f} cov={cov:.2f}"
        + (f" exp={exp:.2f}" if expected_terms else "")
        + f" | chunk={hit.chunk_id} | title={hit.title}\n"
        + f"     {preview}"
    )


def _print_block(
    *,
    label: str,
    hits: list[RAGHit],
    query: str,
    expected_terms: list[str],
) -> None:
    print(f"\n=== {label} (top {len(hits)}) ===")
    if not hits:
        print("No hits.")
        return
    for i, hit in enumerate(hits, start=1):
        print(_format_hit(hit, idx=i, query=query, expected_terms=expected_terms))

    avg_cov = sum(_coverage_score(query, h.text) for h in hits) / len(hits)
    if expected_terms:
        avg_exp = sum(_expected_match_ratio(expected_terms, h.text) for h in hits) / len(hits)
        any_exp = any(_expected_match_ratio(expected_terms, h.text) > 0 for h in hits)
        print(f"Summary: avg_cov={avg_cov:.2f} avg_exp={avg_exp:.2f} expected_hit@{len(hits)}={any_exp}")
    else:
        print(f"Summary: avg_cov={avg_cov:.2f} (provide expected terms to get exp metrics)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive retrieval evaluator for RAG (micro exact + section explain).",
    )
    parser.add_argument("--data-dir", default="data/panaversity_rag_prepared")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-api-key", default="")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--collection-micro", default="panaversity_micro")
    parser.add_argument("--collection-section", default="panaversity_section")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--part-slug", default="")
    parser.add_argument("--chapter-slug", default="")
    parser.add_argument("--lesson-slug", default="")
    args = parser.parse_args()

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

    print("Interactive RAG Eval")
    print("Input format:")
    print("  question")
    print("  question || expected_term1, expected_term2")
    print("Commands: :exit, :quit")
    print(
        f"Filters: part={part_slug or '-'} chapter={chapter_slug or '-'} lesson={lesson_slug or '-'} top_k={args.top_k}"
    )

    while True:
        try:
            raw = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue
        if raw.lower() in {":exit", ":quit", "exit", "quit"}:
            print("Exiting.")
            break

        query, expected_terms = _parse_input(raw)
        if not query:
            continue

        exact_hits = engine.search_exact(
            query,
            part_slug=part_slug,
            chapter_slug=chapter_slug,
            lesson_slug=lesson_slug,
            top_k=args.top_k,
        )
        explain_hits = engine.search_explain(
            query,
            part_slug=part_slug,
            chapter_slug=chapter_slug,
            lesson_slug=lesson_slug,
            top_k=args.top_k,
        )

        print(f"\nQuestion: {query}")
        if expected_terms:
            print(f"Expected terms: {', '.join(expected_terms)}")

        _print_block(label="MICRO / EXACT", hits=exact_hits, query=query, expected_terms=expected_terms)
        _print_block(label="SECTION / EXPLAIN", hits=explain_hits, query=query, expected_terms=expected_terms)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

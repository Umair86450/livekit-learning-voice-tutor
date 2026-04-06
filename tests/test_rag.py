from __future__ import annotations

from livekit_voice_agent.rag import _BM25Index, _ChunkEntry, _tokenize


def test_tokenize_basic() -> None:
    tokens = _tokenize("Hello, Agent Factory! 2026 edition.")
    assert tokens == ["hello", "agent", "factory", "2026", "edition"]


def test_bm25_prefers_relevant_entry() -> None:
    entries = [
        _ChunkEntry(
            chunk_id="c1",
            text="The starter plan costs 29 dollars per month.",
            title="Pricing",
            source_url="https://example.com/pricing",
            chunk_level="micro",
            part_slug=None,
            chapter_slug=None,
            lesson_slug=None,
            section_index=1,
            micro_index=1,
            parent_chunk_id=None,
        ),
        _ChunkEntry(
            chunk_id="c2",
            text="We are SOC 2 compliant and encrypt data at rest.",
            title="Security",
            source_url="https://example.com/security",
            chunk_level="micro",
            part_slug=None,
            chapter_slug=None,
            lesson_slug=None,
            section_index=1,
            micro_index=1,
            parent_chunk_id=None,
        ),
    ]
    bm25 = _BM25Index(entries)
    hits = bm25.search("What is the starter price?", top_k=2)
    assert hits
    best_idx, _score = hits[0]
    assert entries[best_idx].chunk_id == "c1"


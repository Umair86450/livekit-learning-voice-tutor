from __future__ import annotations

import json
import logging
import math
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RAGHit:
    chunk_id: str
    score: float
    source_url: str
    title: str
    text: str
    chunk_level: str
    part_slug: str | None
    chapter_slug: str | None
    lesson_slug: str | None
    section_index: int | None
    micro_index: int | None
    parent_chunk_id: str | None


@dataclass
class _ChunkEntry:
    chunk_id: str
    text: str
    title: str
    source_url: str
    chunk_level: str
    part_slug: str | None
    chapter_slug: str | None
    lesson_slug: str | None
    section_index: int | None
    micro_index: int | None
    parent_chunk_id: str | None


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
            continue
        if len(cur) >= 2:
            out.append("".join(cur))
        cur = []
    if len(cur) >= 2:
        out.append("".join(cur))
    return out


class _BM25Index:
    def __init__(self, entries: list[_ChunkEntry]) -> None:
        self._entries = entries
        self._doc_len: list[int] = []
        self._avg_doc_len = 1.0
        self._term_df: Counter[str] = Counter()
        self._doc_tf: list[Counter[str]] = []

        total_len = 0
        for e in entries:
            toks = _tokenize(e.text)
            tf = Counter(toks)
            self._doc_tf.append(tf)
            dlen = sum(tf.values())
            self._doc_len.append(dlen)
            total_len += dlen
            self._term_df.update(set(tf.keys()))
        if entries:
            self._avg_doc_len = max(1.0, total_len / len(entries))

    def search(
        self,
        query: str,
        *,
        top_k: int,
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
    ) -> list[tuple[int, float]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        n_docs = len(self._entries)
        k1 = 1.2
        b = 0.75
        results: list[tuple[int, float]] = []
        for i, e in enumerate(self._entries):
            if part_slug and e.part_slug != part_slug:
                continue
            if chapter_slug and e.chapter_slug != chapter_slug:
                continue
            if lesson_slug and e.lesson_slug != lesson_slug:
                continue

            tf = self._doc_tf[i]
            dlen = max(1.0, float(self._doc_len[i]))
            score = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                df = self._term_df.get(t, 0)
                idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = f + k1 * (1.0 - b + b * dlen / self._avg_doc_len)
                score += idf * (f * (k1 + 1.0)) / max(1e-9, denom)
            if score > 0:
                results.append((i, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class RAGEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._micro_entries: list[_ChunkEntry] = []
        self._section_entries: list[_ChunkEntry] = []
        self._section_by_chunk_id: dict[str, _ChunkEntry] = {}
        self._bm25_micro: _BM25Index | None = None
        self._bm25_section: _BM25Index | None = None
        self._qdrant_client: Any = None
        self._embedding_model: Any = None
        self._collection_micro = "panaversity_micro"
        self._collection_section = "panaversity_section"
        self._docs_by_id: dict[str, dict[str, Any]] = {}

    @property
    def ready(self) -> bool:
        return self._ready

    def initialize_from_prepared_data(
        self,
        *,
        data_dir: str,
        qdrant_url: str,
        qdrant_api_key: str | None,
        embedding_model_name: str,
        collection_micro: str,
        collection_section: str,
        recreate_collections: bool,
        allow_ingest: bool,
        batch_size: int = 64,
    ) -> None:
        base = Path(data_dir)
        docs_path = base / "docs.jsonl"
        micro_path = base / "chunks_micro.jsonl"
        section_path = base / "chunks_section.jsonl"
        if not docs_path.exists() or not micro_path.exists() or not section_path.exists():
            raise FileNotFoundError(
                f"Prepared RAG files missing in {base}. Expected docs.jsonl, chunks_micro.jsonl, chunks_section.jsonl"
            )

        docs = self._read_jsonl(docs_path)
        micros = self._read_jsonl(micro_path)
        sections = self._read_jsonl(section_path)

        self._docs_by_id = {str(d["doc_id"]): d for d in docs}
        self._micro_entries = [self._to_chunk_entry(r, default_level="micro") for r in micros]
        self._section_entries = [self._to_chunk_entry(r, default_level="section") for r in sections]
        self._section_by_chunk_id = {e.chunk_id: e for e in self._section_entries}
        self._bm25_micro = _BM25Index(self._micro_entries)
        self._bm25_section = _BM25Index(self._section_entries)

        self._collection_micro = collection_micro
        self._collection_section = collection_section
        self._init_qdrant_and_embeddings(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedding_model_name=embedding_model_name,
            recreate_collections=recreate_collections,
            allow_ingest=allow_ingest,
            batch_size=batch_size,
        )
        self._ready = True

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def _to_chunk_entry(self, row: dict[str, Any], *, default_level: str) -> _ChunkEntry:
        return _ChunkEntry(
            chunk_id=str(row.get("chunk_id", "")),
            text=str(row.get("text", "")),
            title=str(row.get("title", "")),
            source_url=str(row.get("url", "")),
            chunk_level=str(row.get("chunk_level", default_level)),
            part_slug=self._clean_opt_str(row.get("part_slug")),
            chapter_slug=self._clean_opt_str(row.get("chapter_slug")),
            lesson_slug=self._clean_opt_str(row.get("lesson_slug")),
            section_index=self._clean_opt_int(row.get("section_index")),
            micro_index=self._clean_opt_int(row.get("micro_index")),
            parent_chunk_id=self._clean_opt_str(row.get("parent_chunk_id")),
        )

    def _clean_opt_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _clean_opt_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return None

    def _init_qdrant_and_embeddings(
        self,
        *,
        qdrant_url: str,
        qdrant_api_key: str | None,
        embedding_model_name: str,
        recreate_collections: bool,
        allow_ingest: bool,
        batch_size: int,
    ) -> None:
        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                HnswConfigDiff,
                PayloadSchemaType,
                PointStruct,
                VectorParams,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Missing RAG dependencies. Install qdrant-client and fastembed."
            ) from exc

        self._embedding_model = TextEmbedding(
            model_name=embedding_model_name,
            providers=["CPUExecutionProvider"],
        )
        self._qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)

        probe = next(self._embedding_model.embed(["dimension probe"]))
        probe_vec = list(probe)
        if not probe_vec:
            raise RuntimeError("Embedding model returned empty vector for probe text.")
        vector_dim = len(probe_vec)

        def ensure_collection(name: str, expected_points: int) -> bool:
            if recreate_collections:
                try:
                    self._qdrant_client.delete_collection(collection_name=name)
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    if self._qdrant_client.collection_exists(collection_name=name):
                        info = self._qdrant_client.get_collection(collection_name=name)
                        points_count = int(info.points_count or 0)
                        if points_count == expected_points:
                            return False
                        if not allow_ingest:
                            raise RuntimeError(
                                f"Collection '{name}' has {points_count} points, expected {expected_points}. "
                                "Run offline ingest first or enable ingest on start."
                            )
                except RuntimeError:
                    raise
                except Exception:  # noqa: BLE001
                    pass
            if not allow_ingest and not recreate_collections:
                raise RuntimeError(
                    f"Collection '{name}' not ready. Run offline ingest first or enable ingest on start."
                )
            self._qdrant_client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE, on_disk=False),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )
            for field in ("doc_id", "part_slug", "chapter_slug", "lesson_slug"):
                try:
                    self._qdrant_client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:  # noqa: BLE001
                    pass
            return True

        need_upsert_micro = ensure_collection(self._collection_micro, expected_points=len(self._micro_entries))
        need_upsert_section = ensure_collection(self._collection_section, expected_points=len(self._section_entries))

        if need_upsert_micro:
            self._upsert_entries(self._collection_micro, self._micro_entries, batch_size=batch_size, PointStruct=PointStruct)
        if need_upsert_section:
            self._upsert_entries(self._collection_section, self._section_entries, batch_size=batch_size, PointStruct=PointStruct)

    def _upsert_entries(self, collection: str, entries: list[_ChunkEntry], *, batch_size: int, PointStruct: Any) -> None:
        points: list[Any] = []
        point_id = 1
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            vectors = list(self._embedding_model.embed([e.text for e in batch]))
            for e, vec in zip(batch, vectors, strict=False):
                payload = {
                    "chunk_id": e.chunk_id,
                    "doc_id": e.chunk_id.split("-s", 1)[0],
                    "url": e.source_url,
                    "title": e.title,
                    "text": e.text,
                    "chunk_level": e.chunk_level,
                    "part_slug": e.part_slug,
                    "chapter_slug": e.chapter_slug,
                    "lesson_slug": e.lesson_slug,
                    "section_index": e.section_index,
                    "micro_index": e.micro_index,
                    "parent_chunk_id": e.parent_chunk_id,
                }
                points.append(PointStruct(id=point_id, vector=list(vec), payload=payload))
                point_id += 1
            self._qdrant_client.upsert(collection_name=collection, points=points, wait=True)
            points = []

    def _embed_query(self, query: str) -> list[float]:
        vecs = list(self._embedding_model.embed([query]))
        return list(vecs[0]) if vecs else []

    def _qdrant_filter(self, *, part_slug: str | None, chapter_slug: str | None, lesson_slug: str | None) -> Any:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except Exception:  # noqa: BLE001
            return None
        conditions: list[Any] = []
        if part_slug:
            conditions.append(FieldCondition(key="part_slug", match=MatchValue(value=part_slug)))
        if chapter_slug:
            conditions.append(FieldCondition(key="chapter_slug", match=MatchValue(value=chapter_slug)))
        if lesson_slug:
            conditions.append(FieldCondition(key="lesson_slug", match=MatchValue(value=lesson_slug)))
        if not conditions:
            return None
        return Filter(must=conditions)

    def _vector_search(
        self,
        *,
        collection: str,
        query: str,
        top_k: int,
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
    ) -> list[RAGHit]:
        if not self._qdrant_client or not self._embedding_model:
            return []
        qv = self._embed_query(query)
        if not qv:
            return []
        q_filter = self._qdrant_filter(part_slug=part_slug, chapter_slug=chapter_slug, lesson_slug=lesson_slug)
        try:
            if hasattr(self._qdrant_client, "search"):
                points = self._qdrant_client.search(
                    collection_name=collection,
                    query_vector=qv,
                    query_filter=q_filter,
                    with_payload=True,
                    limit=top_k,
                )
            else:
                response = self._qdrant_client.query_points(
                    collection_name=collection,
                    query=qv,
                    query_filter=q_filter,
                    with_payload=True,
                    limit=top_k,
                )
                points = getattr(response, "points", []) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search failed: %s", exc)
            return []
        out: list[RAGHit] = []
        for p in points:
            payload = p.payload or {}
            out.append(
                RAGHit(
                    chunk_id=str(payload.get("chunk_id", "")),
                    score=float(p.score or 0.0),
                    source_url=str(payload.get("url", "")),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                    chunk_level=str(payload.get("chunk_level", "")),
                    part_slug=self._clean_opt_str(payload.get("part_slug")),
                    chapter_slug=self._clean_opt_str(payload.get("chapter_slug")),
                    lesson_slug=self._clean_opt_str(payload.get("lesson_slug")),
                    section_index=self._clean_opt_int(payload.get("section_index")),
                    micro_index=self._clean_opt_int(payload.get("micro_index")),
                    parent_chunk_id=self._clean_opt_str(payload.get("parent_chunk_id")),
                )
            )
        return out

    def _bm25_search_hits(
        self,
        *,
        query: str,
        entries: list[_ChunkEntry],
        bm25: _BM25Index | None,
        top_k: int,
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
    ) -> list[RAGHit]:
        if bm25 is None:
            return []
        idx_scored = bm25.search(
            query,
            top_k=top_k,
            part_slug=part_slug,
            chapter_slug=chapter_slug,
            lesson_slug=lesson_slug,
        )
        return [
            RAGHit(
                chunk_id=entries[i].chunk_id,
                score=score,
                source_url=entries[i].source_url,
                title=entries[i].title,
                text=entries[i].text,
                chunk_level=entries[i].chunk_level,
                part_slug=entries[i].part_slug,
                chapter_slug=entries[i].chapter_slug,
                lesson_slug=entries[i].lesson_slug,
                section_index=entries[i].section_index,
                micro_index=entries[i].micro_index,
                parent_chunk_id=entries[i].parent_chunk_id,
            )
            for i, score in idx_scored
        ]

    def _merge_hybrid(self, vector_hits: list[RAGHit], bm25_hits: list[RAGHit], *, top_k: int) -> list[RAGHit]:
        def normalize(hits: list[RAGHit]) -> dict[str, float]:
            if not hits:
                return {}
            max_s = max(h.score for h in hits) or 1.0
            return {h.chunk_id: h.score / max_s for h in hits}

        v_norm = normalize(vector_hits)
        b_norm = normalize(bm25_hits)
        merged: dict[str, RAGHit] = {}
        for h in vector_hits + bm25_hits:
            merged[h.chunk_id] = h
        scored: list[tuple[RAGHit, float]] = []
        for cid, h in merged.items():
            final_score = 0.65 * v_norm.get(cid, 0.0) + 0.35 * b_norm.get(cid, 0.0)
            scored.append((h, final_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RAGHit(**{**h.__dict__, "score": s}) for h, s in scored[:top_k]]

    def search_exact(
        self,
        query: str,
        *,
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
        top_k: int = 5,
    ) -> list[RAGHit]:
        with self._lock:
            if not self._ready:
                return []
            vector_hits = self._vector_search(
                collection=self._collection_micro,
                query=query,
                top_k=max(top_k * 2, 10),
                part_slug=part_slug,
                chapter_slug=chapter_slug,
                lesson_slug=lesson_slug,
            )
            bm25_hits = self._bm25_search_hits(
                query=query,
                entries=self._micro_entries,
                bm25=self._bm25_micro,
                top_k=max(top_k * 2, 10),
                part_slug=part_slug,
                chapter_slug=chapter_slug,
                lesson_slug=lesson_slug,
            )
            return self._merge_hybrid(vector_hits, bm25_hits, top_k=top_k)

    def search_explain(
        self,
        query: str,
        *,
        part_slug: str | None = None,
        chapter_slug: str | None = None,
        lesson_slug: str | None = None,
        top_k: int = 3,
    ) -> list[RAGHit]:
        with self._lock:
            if not self._ready:
                return []
            vector_hits = self._vector_search(
                collection=self._collection_section,
                query=query,
                top_k=max(top_k * 2, 8),
                part_slug=part_slug,
                chapter_slug=chapter_slug,
                lesson_slug=lesson_slug,
            )
            bm25_hits = self._bm25_search_hits(
                query=query,
                entries=self._section_entries,
                bm25=self._bm25_section,
                top_k=max(top_k * 2, 8),
                part_slug=part_slug,
                chapter_slug=chapter_slug,
                lesson_slug=lesson_slug,
            )
            return self._merge_hybrid(vector_hits, bm25_hits, top_k=top_k)

    def expand_micro_to_sections(self, micro_hits: list[RAGHit], *, max_sections: int = 2) -> list[RAGHit]:
        seen: set[str] = set()
        out: list[RAGHit] = []
        with self._lock:
            for h in micro_hits:
                if not h.parent_chunk_id:
                    continue
                if h.parent_chunk_id in seen:
                    continue
                seen.add(h.parent_chunk_id)
                sec = self._section_by_chunk_id.get(h.parent_chunk_id)
                if not sec:
                    continue
                out.append(
                    RAGHit(
                        chunk_id=sec.chunk_id,
                        score=h.score,
                        source_url=sec.source_url,
                        title=sec.title,
                        text=sec.text,
                        chunk_level=sec.chunk_level,
                        part_slug=sec.part_slug,
                        chapter_slug=sec.chapter_slug,
                        lesson_slug=sec.lesson_slug,
                        section_index=sec.section_index,
                        micro_index=sec.micro_index,
                        parent_chunk_id=sec.parent_chunk_id,
                    )
                )
                if len(out) >= max_sections:
                    break
        return out

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    return WS_RE.sub(" ", text).strip()


def words_from_text(text: str) -> list[str]:
    return [w for w in normalize_text(text).split(" ") if w]


def split_into_paragraphs(text: str) -> list[str]:
    raw = text.replace("\r\n", "\n")
    parts = re.split(r"\n\s*\n+", raw)
    out: list[str] = []
    for p in parts:
        p2 = normalize_text(p)
        if p2:
            out.append(p2)
    return out


def chunk_paragraphs(
    paragraphs: list[str],
    *,
    target_words: int,
    max_words: int,
) -> list[dict[str, Any]]:
    normalized_paras: list[tuple[str, int, int]] = []
    for i, para in enumerate(paragraphs):
        pw = words_from_text(para)
        if len(pw) <= max_words:
            normalized_paras.append((para, i, i))
            continue
        # Oversized paragraph fallback: split by max_words windows.
        start = 0
        while start < len(pw):
            end = min(len(pw), start + max_words)
            normalized_paras.append((" ".join(pw[start:end]), i, i))
            if end >= len(pw):
                break
            start = end

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_words = 0
    para_start = 0
    para_end = 0

    for para, src_start, src_end in normalized_paras:
        para_words = len(words_from_text(para))
        if not current:
            current = [para]
            current_words = para_words
            para_start = src_start
            para_end = src_end
            continue

        if current_words < target_words and current_words + para_words <= max_words:
            current.append(para)
            current_words += para_words
            para_end = src_end
            continue

        chunks.append(
            {
                "text": "\n\n".join(current),
                "paragraph_start": para_start,
                "paragraph_end": para_end,
            }
        )
        current = [para]
        current_words = para_words
        para_start = src_start
        para_end = src_end

    if current:
        chunks.append(
            {
                "text": "\n\n".join(current),
                "paragraph_start": para_start,
                "paragraph_end": para_end,
            }
        )
    return chunks


def chunk_words_sliding(
    text: str,
    *,
    target_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    words = words_from_text(text)
    if not words:
        return []
    if target_words <= 0:
        target_words = 1
    if overlap_words >= target_words:
        overlap_words = max(0, target_words // 5)

    chunks: list[dict[str, Any]] = []
    start = 0
    step = max(1, target_words - overlap_words)

    while start < len(words):
        end = min(len(words), start + target_words)
        piece = words[start:end]
        if not piece:
            break
        chunks.append(
            {
                "text": " ".join(piece),
                "word_start": start,
                "word_end": end - 1,
            }
        )
        if end >= len(words):
            break
        start += step
    return chunks


def to_doc_id(sequence: int, url: str) -> str:
    path = url.split("/docs/", 1)[-1].strip("/")
    slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-") or "about"
    return f"{sequence:04d}-{slug}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def prepare(input_dir: Path, output_dir: Path) -> dict[str, int]:
    index = read_json(input_dir / "docs_index.json")
    records: list[dict[str, Any]] = index.get("records", [])

    docs_out: list[dict[str, Any]] = []
    section_chunks: list[dict[str, Any]] = []
    micro_chunks: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        sequence = int(rec["sequence"])
        doc_id = to_doc_id(sequence, rec["url"])
        prev_doc_id = to_doc_id(int(records[i - 1]["sequence"]), records[i - 1]["url"]) if i > 0 else None
        next_doc_id = (
            to_doc_id(int(records[i + 1]["sequence"]), records[i + 1]["url"])
            if i + 1 < len(records)
            else None
        )

        text_file = input_dir / rec["text_file"]
        raw_text = text_file.read_text(encoding="utf-8") if text_file.exists() else ""
        clean_text = raw_text.strip()
        paragraphs = split_into_paragraphs(clean_text)
        if not paragraphs and clean_text:
            paragraphs = [normalize_text(clean_text)]

        docs_out.append(
            {
                "doc_id": doc_id,
                "sequence": sequence,
                "url": rec["url"],
                "title": rec.get("title"),
                "part_slug": rec.get("part_slug"),
                "chapter_slug": rec.get("chapter_slug"),
                "lesson_slug": rec.get("lesson_slug"),
                "prev_doc_id": prev_doc_id,
                "next_doc_id": next_doc_id,
            }
        )

        # Section-level chunking: ~500-800 tokens => ~375-600 words.
        sec_blocks = chunk_paragraphs(
            paragraphs,
            target_words=460,
            max_words=600,
        )
        for sec_idx, sec in enumerate(sec_blocks, start=1):
            section_id = f"{doc_id}-s{sec_idx:03d}"
            sec_row = {
                "chunk_id": section_id,
                "chunk_level": "section",
                "doc_id": doc_id,
                "sequence": sequence,
                "url": rec["url"],
                "title": rec.get("title"),
                "part_slug": rec.get("part_slug"),
                "chapter_slug": rec.get("chapter_slug"),
                "lesson_slug": rec.get("lesson_slug"),
                "section_index": sec_idx,
                "text": sec["text"],
            }
            section_chunks.append(sec_row)

            # Micro-chunking: ~180-260 tokens => ~135-195 words.
            micros = chunk_words_sliding(
                sec["text"],
                target_words=165,
                overlap_words=28,
            )
            for micro_idx, micro in enumerate(micros, start=1):
                micro_chunks.append(
                    {
                        "chunk_id": f"{section_id}-m{micro_idx:03d}",
                        "chunk_level": "micro",
                        "parent_chunk_id": section_id,
                        "doc_id": doc_id,
                        "sequence": sequence,
                        "url": rec["url"],
                        "title": rec.get("title"),
                        "part_slug": rec.get("part_slug"),
                        "chapter_slug": rec.get("chapter_slug"),
                        "lesson_slug": rec.get("lesson_slug"),
                        "section_index": sec_idx,
                        "micro_index": micro_idx,
                        "text": micro["text"],
                    }
                )

    all_chunks = sorted(
        [*section_chunks, *micro_chunks],
        key=lambda r: (r["sequence"], r["chunk_id"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", {
        "source_dir": str(input_dir),
        "total_docs": len(docs_out),
        "total_section_chunks": len(section_chunks),
        "total_micro_chunks": len(micro_chunks),
        "total_chunks": len(all_chunks),
        "strategy": {
            "section_tokens_target": "500-800",
            "section_words_target": 460,
            "section_words_max": 600,
            "micro_tokens_target": "180-260",
            "micro_words_target": 165,
            "micro_words_overlap": 28,
        },
    })
    write_jsonl(output_dir / "docs.jsonl", docs_out)
    write_jsonl(output_dir / "chunks_section.jsonl", section_chunks)
    write_jsonl(output_dir / "chunks_micro.jsonl", micro_chunks)
    write_jsonl(output_dir / "chunks_all.jsonl", all_chunks)

    return {
        "docs": len(docs_out),
        "section_chunks": len(section_chunks),
        "micro_chunks": len(micro_chunks),
        "all_chunks": len(all_chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Panaversity scrape into RAG-ready chunked dataset.")
    parser.add_argument(
        "--input-dir",
        default="data/panaversity_scrape_playwright_full",
        help="Input scrape folder containing docs_index.json and content/*.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="data/panaversity_rag_prepared",
        help="Output folder for prepared docs/chunks JSONL files.",
    )
    args = parser.parse_args()

    counts = prepare(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
    print(
        f"Prepared docs={counts['docs']} section_chunks={counts['section_chunks']} "
        f"micro_chunks={counts['micro_chunks']} total_chunks={counts['all_chunks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

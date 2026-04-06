# Pattern Blueprints (LangChain + Qdrant)

This file maps advanced patterns to concrete implementation choices using official docs.

## 1) Semantic RAG (default)
Use when:
- requirements are standard QA over stable corpus
- low complexity and predictable latency are priorities

Blueprint:
1. ingest chunks + embeddings into Qdrant
2. build retriever from vector store
3. compose prompt + model chain
4. enforce unknown-answer fallback
5. return citations/provenance

## 2) HyDE-Style Retrieval
Use when:
- user queries are short, vague, or domain-jargon heavy
- baseline dense retrieval misses relevant docs

Blueprint:
1. generate hypothetical answer/document from query
2. embed hypothetical text
3. retrieve with that embedding
4. answer only from retrieved real documents

Note:
- HyDE procedure is an inferred pattern built on official LangChain runnable/query-rewrite capabilities.

## 3) Corrective Retrieval (CRAG-Style)
Use when:
- retrieval quality is inconsistent across query types
- you need rewrite/fallback branch when context is weak

Blueprint:
1. initial retrieval
2. quality gate (heuristic/model-based)
3. if weak: rewrite query or broaden retrieval
4. re-retrieve and answer

Note:
- CRAG-style branching is implemented via official workflow graph/query rewrite patterns.

## 4) Agentic RAG
Use when:
- retrieval is not always needed
- multiple tools/sources must be selected dynamically
- iterative retrieve-think-act cycles are required

Blueprint:
1. expose retrieval as tool
2. let agent decide when/what to retrieve
3. preserve retrieved artifacts for provenance
4. enforce guardrail prompt: no unsupported claims

## 5) Validation Matrix
- retrieval relevance (top-k quality)
- grounding/citation accuracy
- unknown-answer correctness rate
- latency budget per query path
- failure behavior under empty/noisy retrieval

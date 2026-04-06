---
name: rag-production-pipelines
description: Build production Retrieval-Augmented Generation (RAG) pipelines with LangChain and Qdrant using official documentation only. Use this skill when deciding RAG vs LLM-only approaches, and when implementing or upgrading semantic retrieval, multi-query retrieval, HyDE-style query expansion, corrective (CRAG-style) workflows, or agentic RAG with tool-driven retrieval decisions, plus evaluation and reliability checks.
---

# RAG Production Pipelines

## Overview
Use this skill to design and implement production-grade RAG with LangChain + Qdrant, from basic semantic search to advanced retrieval patterns. Ground implementation decisions in official docs referenced in `references/` and avoid assumptions.

## Core Concept: Retrieve -> Augment -> Generate
- Retrieve: fetch relevant external context from a trusted knowledge store.
- Augment: inject retrieved evidence into the model prompt with source metadata/citations.
- Generate: produce an answer constrained by retrieved evidence, with unknown fallback when evidence is insufficient.
- Use this pattern whenever correctness depends on data outside model parametric memory.

## RAG Necessity Checklist (RAG vs LLM-Only)
Use this checklist before architecture decisions:
- Knowledge cutoff risk: required facts may be newer than model training data.
- Domain-specific/private data: answers depend on proprietary docs, internal policy, or tenant-specific records.
- Verifiability requirement: output must include citations, provenance, or auditable evidence.
- High-cost hallucination risk: incorrect answers create legal, financial, or operational harm.
- Rapidly changing content: product docs, prices, policies, or runbooks change frequently.
- Long-tail factual queries: exact IDs, clauses, configurations, or release-note details are common.

Decision rule:
- If 2+ checklist items are true, default to RAG (or hybrid with retrieval fallback).
- If 0-1 items are true and task is general reasoning, LLM-only may be sufficient.

## Warning Signs LLM-Only Will Hallucinate
- The model answers confidently but cannot provide sourceable evidence.
- User asks for exact, recent, or compliance-sensitive facts.
- Similar questions return inconsistent factual outputs.
- Prompt requires organization-specific knowledge not in public training data.
- Failures cluster around entity names, versions, IDs, or policy wording.
- Model fabricates references, endpoints, or document titles.

## Required Workflow
1. Confirm goal and constraints.
2. Read official references before coding.
3. Choose the minimal RAG pattern that meets requirements.
4. Implement retrieval and answer chain with citations.
5. Add evaluation and failure handling.
6. Verify with deterministic checks before finalizing.

## Step 1: Confirm Goal and Constraints
Capture these inputs first:
- corpus source and update frequency
- latency/throughput targets
- recall vs precision priority
- multi-tenant/filtering requirements
- citation requirements
- failure policy when no evidence is retrieved

If any item is missing, state assumptions explicitly before implementation.

## Step 2: Read Official References
Read these files first:
- `references/langchain-official.md`
- `references/qdrant-official.md`

Read this file when selecting advanced patterns:
- `references/pattern-blueprints.md`

## Step 3: Pattern Selection
Choose the lightest pattern that solves the problem:
- Semantic RAG: default baseline for most knowledge QA.
- Multi-query retrieval: when recall is weak for underspecified queries.
- HyDE-style: when queries are short/ambiguous and need better dense retrieval anchors.
- Corrective (CRAG-style): when retrieval quality varies and query rewriting/fallback is required.
- Agentic RAG: when model must decide whether/when to retrieve, route tools, or iteratively refine.

## Step 4: Implementation Rules
- Build retriever from vector store (`as_retriever`) and compose with a prompt + model chain.
- Enforce "answer from context or say unknown" behavior.
- Keep retrieved documents as artifacts/metadata for traceability and citations.
- Prefer payload-filtered retrieval for tenant/domain constraints.
- Keep chunking/indexing aligned with retrieval strategy and query type.

## Step 5: Production Hardening
- Add payload indexes for frequently filtered fields.
- Configure hybrid or filtered search when metadata constraints matter.
- Add monitoring for retrieval hit quality and empty-context rate.
- Add retry/fallback strategy for retrieval/tool failures.
- Add regression tests for retrieval relevance and answer grounding.

## Step 6: Verification
Run deterministic checks:
- project tests covering changed retrieval behavior
- focused RAG tests (`tests/test_rag.py` in this repo)
- linter and formatting checks
- optional local preflight script: `scripts/rag_smoke_check.sh`

## Output Contract
When using this skill, return:
1. chosen pattern and why
2. retrieval architecture summary
3. code/config changes
4. verification commands + results
5. known limits and next hardening steps

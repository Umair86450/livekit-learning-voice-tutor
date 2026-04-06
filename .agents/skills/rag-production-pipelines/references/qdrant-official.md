# Qdrant Official Notes (Context7)

Sources:
- https://qdrant.tech/documentation/tutorials-build-essentials/rag-deepseek
- https://qdrant.tech/documentation/examples/llama-index-multitenancy
- https://qdrant.tech/documentation/manage-data/indexing
- https://qdrant.tech/documentation/concepts/indexing
- https://qdrant.tech/documentation/examples/hybrid-search-llamaindex-jinaai

## Baseline RAG with Qdrant
- Query relevant points from collection and build answer prompt from retrieved payload context.
- Enforce unknown-answer behavior when context does not support response.

## Production Collection Practices
- Create payload indexes for fields used in filters (for speed and predictable latency).
- Use metadata filtering for tenant/library/category scoping.
- Tune collection/HNSW settings for workload profile.

## Search Modes
- Dense semantic retrieval is baseline.
- Hybrid search is supported and useful when lexical + semantic signals both matter.

## Indexing Guidance
- Text indexes can be configured for payload fields.
- Tokenizer/case settings should match query behavior and language assumptions.

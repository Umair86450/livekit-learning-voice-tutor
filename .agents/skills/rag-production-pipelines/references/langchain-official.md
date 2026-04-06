# LangChain Official Notes (Context7)

Sources:
- https://docs.langchain.com/oss/python/integrations/vectorstores/teradata
- https://docs.langchain.com/oss/python/integrations/vectorstores/jaguar
- https://docs.langchain.com/oss/python/langchain/rag
- https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow
- https://docs.langchain.com/oss/python/integrations/chat/anthropic

## Baseline RAG (official composition)
- Use retriever + prompt + model + output parser runnable pipeline.
- Convert vector store to retriever via `as_retriever(...)`.
- Keep prompt explicit: answer only from provided context; say unknown when context is insufficient.

## Agentic RAG (official patterns)
- Define retrieval as a tool (`@tool`) and return both serialized content and artifacts when needed.
- Use workflow/graph composition to separate:
  - query rewrite
  - retrieval
  - answer generation/tool use
- Agentic flow is appropriate when model must decide retrieval/tool routing dynamically.

## Retrieval Quality Enhancers (officially represented)
- Multi-query retrieval pattern is documented for recall improvements.
- Query rewriting workflows are documented via custom graph pipelines.

## Practical Guidance
- Start with baseline RAG first, then layer multi-query/rewrite/agentic control only if metrics justify complexity.
- Keep retrieved docs available for citation/provenance.

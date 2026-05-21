"""
Retriever: turns a natural-language query into a list of RetrievedChunks.

Pipeline:
    query → embed → vector search → (optional rerank) → top-k chunks
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from ingestion.embed import embed_one
from retrieval.vector_store import RetrievedChunk, VectorStore

load_dotenv()

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_STRATEGY = os.getenv("DEFAULT_CHUNKING_STRATEGY", "fixed_size")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = DEFAULT_STRATEGY,
    doc_type: str | None = None,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    own_store = store is None
    if own_store:
        store = VectorStore()

    try:
        q_vec = embed_one(query)

        if RERANK_ENABLED:
            # Pull a wider pool, then rerank down to top_k
            candidates = store.search(q_vec, top_k=RERANK_TOP_N,
                                      strategy=strategy, doc_type=doc_type)
            from retrieval.rerank import rerank
            return rerank(query, candidates, top_k=top_k)

        return store.search(q_vec, top_k=top_k, strategy=strategy, doc_type=doc_type)
    finally:
        if own_store:
            store.close()


# CLI for quick smoke-testing
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "how do I trigger a workflow with a webhook"
    print(f"Query: {q}\n")
    for r in retrieve(q):
        print(f"[{r.similarity:.3f}] {r.title}")
        print(f"   {r.source_url}")
        print(f"   {r.content[:200]}...\n")

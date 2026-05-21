"""
Reranker layer (day 2).

Two-stage retrieval:
  1. Vector search returns top-N candidates (e.g. N=20).
  2. A reranker re-scores each candidate against the query directly,
     usually with a cross-encoder, and keeps the top_k.

Cross-encoders are slower but more accurate than bi-encoder (vector) similarity
because they attend across the query and candidate jointly.

Implementations:
- Cohere `rerank-3` (managed API, fast, free tier available).
- Local cross-encoder (sentence-transformers) — no API key needed.

Day 2 task: pick one, wire it in, re-run eval, record the delta in DECISIONS.md.
"""

from __future__ import annotations

import os
from typing import Any

from retrieval.vector_store import RetrievedChunk


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Pick ONE of the implementations below and return the top_k reranked chunks.
    For day 1, this is a passthrough.
    """
    return candidates[:top_k]


# --- Implementation A: Cohere rerank-3 -----------------------------------
#
# Requires: pip install cohere
#           COHERE_API_KEY in .env
#
# def rerank(query, candidates, top_k=5):
#     import cohere
#     co = cohere.Client(os.getenv("COHERE_API_KEY"))
#     resp = co.rerank(
#         model="rerank-3",
#         query=query,
#         documents=[c.content for c in candidates],
#         top_n=top_k,
#     )
#     return [candidates[r.index] for r in resp.results]


# --- Implementation B: local cross-encoder -------------------------------
#
# Requires: pip install sentence-transformers
#
# from sentence_transformers import CrossEncoder
#
# _model: CrossEncoder | None = None
#
# def rerank(query, candidates, top_k=5):
#     global _model
#     if _model is None:
#         _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     pairs = [(query, c.content) for c in candidates]
#     scores = _model.predict(pairs)
#     ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
#     return [c for _, c in ranked[:top_k]]

"""
Retrieval evaluation metrics.

Given a query, a list of retrieved chunk IDs (or source URLs), and a set of
ground-truth relevant items, compute:

- precision@k:  fraction of top-k that are relevant
- recall@k:     fraction of all relevant items that appear in top-k
- MRR:          mean reciprocal rank of the first relevant item

The eval set in data/eval/eval_set.jsonl uses source URLs as the unit of
ground truth (each query lists one or more "relevant_urls"). We then check if
any retrieved chunk's source_url is in the relevant set.
"""

from __future__ import annotations

from statistics import mean


def precision_at_k(retrieved_urls: list[str], relevant_urls: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_urls[:k]
    if not top:
        return 0.0
    hits = sum(1 for u in top if u in relevant_urls)
    return hits / len(top)


def recall_at_k(retrieved_urls: list[str], relevant_urls: set[str], k: int) -> float:
    if not relevant_urls:
        return 0.0
    top = set(retrieved_urls[:k])
    hits = len(top & relevant_urls)
    return hits / len(relevant_urls)


def reciprocal_rank(retrieved_urls: list[str], relevant_urls: set[str]) -> float:
    for i, url in enumerate(retrieved_urls, start=1):
        if url in relevant_urls:
            return 1.0 / i
    return 0.0


def aggregate(per_query: list[dict]) -> dict:
    """Mean each metric across the eval set."""
    if not per_query:
        return {}
    return {
        k: round(mean(q[k] for q in per_query), 4)
        for k in per_query[0].keys()
        if isinstance(per_query[0][k], (int, float))
    }

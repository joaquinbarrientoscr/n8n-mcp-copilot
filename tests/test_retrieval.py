"""
Smoke test — verifies the basics of the pipeline.

Run with:
    pytest tests/test_retrieval.py -v

These tests require:
- Postgres running (`docker compose up -d`)
- At least some data ingested (`python -m ingestion.ingest`)
- OPENAI_API_KEY set (for query embedding)

If those aren't set up, the tests skip rather than fail.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="module")
def store():
    try:
        from retrieval.vector_store import VectorStore
        s = VectorStore()
        yield s
        s.close()
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"DB unavailable: {e}")


def test_database_has_data(store):
    stats = store.stats()
    if stats["documents"] == 0:
        pytest.skip("No documents ingested yet — run `python -m ingestion.ingest`")
    assert stats["chunks_total"] > 0


def test_retrieve_returns_results(store):
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    from retrieval.retriever import retrieve
    results = retrieve("how do I create a webhook trigger in n8n", top_k=5)
    assert len(results) > 0
    assert all(0.0 <= r.similarity <= 1.0 for r in results)


def test_chunking_fixed_size():
    from ingestion.chunk import chunk_fixed_size
    text = "Sentence one. " * 500
    chunks = chunk_fixed_size(text)
    assert len(chunks) > 1
    assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)

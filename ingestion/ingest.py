"""
End-to-end ingestion pipeline.

    docs sources → scrape → chunk → embed → Postgres+pgvector

Usage:
    python -m ingestion.ingest                       # default: fixed_size
    python -m ingestion.ingest --strategy semantic   # alternative strategy
    python -m ingestion.ingest --reset               # wipe DB first

After ingestion, both `documents` and `chunks` tables are populated.
You can ingest the same corpus with multiple strategies — the
`chunking_strategy` column distinguishes them, so eval can compare.
"""

from __future__ import annotations

import argparse
import os
from itertools import chain

from dotenv import load_dotenv

from ingestion.chunk import chunk as do_chunk
from ingestion.embed import embed_texts
from ingestion.scrape import ScrapedDoc, scrape_mcp, scrape_n8n
from retrieval.vector_store import VectorStore

load_dotenv()

DEFAULT_STRATEGY = os.getenv("DEFAULT_CHUNKING_STRATEGY", "fixed_size")


def _iter_all_docs():
    yield from chain(scrape_n8n(), scrape_mcp())


def run(strategy: str = DEFAULT_STRATEGY, reset: bool = False) -> None:
    store = VectorStore()

    if reset:
        print("⚠ Wiping documents and chunks tables…")
        store.reset()

    print(f"Strategy: {strategy}")
    print("─" * 50)

    total_docs = 0
    total_chunks = 0
    pending_texts: list[str] = []
    pending_meta: list[tuple[int, int, str]] = []        # (doc_id, chunk_idx, text)

    for doc in _iter_all_docs():
        doc_id = store.upsert_document(
            source_url=doc.source_url,
            title=doc.title,
            doc_type=doc.doc_type,
            raw_content=doc.content,
        )
        total_docs += 1

        # Avoid re-chunking the same (doc, strategy) pair
        if store.has_chunks(doc_id, strategy):
            continue

        pieces = do_chunk(doc.content, strategy=strategy)
        for idx, text in pieces:
            pending_texts.append(text)
            pending_meta.append((doc_id, idx, text))

        # Flush in batches to keep memory bounded
        if len(pending_texts) >= 500:
            _flush(store, pending_texts, pending_meta, strategy)
            total_chunks += len(pending_texts)
            pending_texts.clear()
            pending_meta.clear()

    if pending_texts:
        _flush(store, pending_texts, pending_meta, strategy)
        total_chunks += len(pending_texts)

    print("─" * 50)
    print(f"✓ Ingested {total_docs} documents → {total_chunks} chunks "
          f"(strategy={strategy})")


def _flush(
    store: VectorStore,
    texts: list[str],
    meta: list[tuple[int, int, str]],
    strategy: str,
) -> None:
    vectors = embed_texts(texts)
    rows = [
        (doc_id, chunk_idx, content, vec, strategy, {})
        for (doc_id, chunk_idx, content), vec in zip(meta, vectors)
    ]
    store.insert_chunks(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY,
                        choices=["fixed_size", "semantic"])
    parser.add_argument("--reset", action="store_true",
                        help="Wipe documents+chunks before ingesting")
    args = parser.parse_args()
    run(strategy=args.strategy, reset=args.reset)

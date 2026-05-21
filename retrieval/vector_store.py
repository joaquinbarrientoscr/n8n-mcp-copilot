"""
Postgres + pgvector wrapper.

Keeps SQL in one place so the rest of the codebase doesn't carry DB concerns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    content: str
    source_url: str
    title: str
    doc_type: str
    similarity: float            # cosine similarity (higher = closer)
    chunk_index: int


class VectorStore:
    def __init__(self) -> None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set in environment")
        self.conn = psycopg2.connect(DATABASE_URL)
        register_vector(self.conn)

    # ----- writes -----
    def reset(self) -> None:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE;")

    def upsert_document(
        self, source_url: str, title: str, doc_type: str, raw_content: str
    ) -> int:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source_url, title, doc_type, raw_content)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_url) DO UPDATE
                  SET title = EXCLUDED.title,
                      raw_content = EXCLUDED.raw_content,
                      scraped_at = NOW()
                RETURNING id
                """,
                (source_url, title, doc_type, raw_content),
            )
            return cur.fetchone()[0]

    def has_chunks(self, document_id: int, strategy: str) -> bool:
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chunks WHERE document_id=%s AND chunking_strategy=%s LIMIT 1",
                (document_id, strategy),
            )
            return cur.fetchone() is not None

    def insert_chunks(
        self,
        rows: list[tuple[int, int, str, list[float], str, dict[str, Any]]],
    ) -> None:
        """rows: (document_id, chunk_index, content, embedding, strategy, metadata)"""
        with self.conn, self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO chunks
                    (document_id, chunk_index, content, embedding,
                     chunking_strategy, metadata)
                VALUES %s
                """,
                [
                    (doc_id, idx, content, vec, strategy, json.dumps(meta))
                    for (doc_id, idx, content, vec, strategy, meta) in rows
                ],
            )

    # ----- reads -----
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        strategy: str | None = "fixed_size",
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Cosine similarity search; optionally filter by chunking strategy / doc_type."""
        filters = []
        params: list[Any] = [query_vector]
        if strategy:
            filters.append("c.chunking_strategy = %s")
            params.append(strategy)
        if doc_type:
            filters.append("d.doc_type = %s")
            params.append(doc_type)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        # 1 - (embedding <=> query) gives cosine similarity in [-1, 1].
        # pgvector's `<=>` returns cosine distance, lower is closer.
        sql = f"""
            SELECT c.id, c.document_id, c.content, d.source_url, d.title, d.doc_type,
                   1 - (c.embedding <=> %s::vector) AS similarity,
                   c.chunk_index
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        params = [query_vector] + params[1:] + [query_vector, top_k]

        with self.conn, self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                chunk_id=r[0], document_id=r[1], content=r[2],
                source_url=r[3], title=r[4] or "", doc_type=r[5],
                similarity=float(r[6]), chunk_index=r[7],
            )
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        with self.conn, self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            n_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunks")
            n_chunks = cur.fetchone()[0]
            cur.execute(
                "SELECT chunking_strategy, COUNT(*) FROM chunks GROUP BY 1"
            )
            by_strategy = dict(cur.fetchall())
        return {
            "documents": n_docs,
            "chunks_total": n_chunks,
            "chunks_by_strategy": by_strategy,
        }

    def close(self) -> None:
        self.conn.close()

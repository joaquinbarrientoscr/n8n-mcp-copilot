"""
Embedding generation using OpenAI's embedding models.

Batched to amortize API overhead. text-embedding-3-small at 1536d is the
default — see DECISIONS.md (D-002).
"""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = 100        # OpenAI allows up to 2048 inputs but 100 is conservative

_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_texts(texts: list[str], show_progress: bool = True) -> list[list[float]]:
    """Embed a list of strings; returns a list of vectors in the same order."""
    client = _client_singleton()
    out: list[list[float]] = []

    batches = range(0, len(texts), BATCH_SIZE)
    iterator = tqdm(batches, desc="Embedding", disable=not show_progress)

    for start in iterator:
        batch = texts[start : start + BATCH_SIZE]
        # Replace empty strings to avoid API error
        batch = [t if t.strip() else " " for t in batch]
        resp = client.embeddings.create(model=MODEL, input=batch)
        out.extend(item.embedding for item in resp.data)

    return out


def embed_one(text: str) -> list[float]:
    """Convenience for single-query embedding (used at retrieval time)."""
    return embed_texts([text], show_progress=False)[0]

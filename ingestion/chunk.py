"""
Chunking strategies.

Two strategies implemented:
- fixed_size: token-aware splitter with overlap (default, fast, predictable).
- semantic:   semantic boundary detection (LangChain SemanticChunker).
              Higher quality, slower at ingest time.

Both yield a list of (chunk_index, text) tuples per document.
"""

from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))


def chunk_fixed_size(text: str) -> list[tuple[int, str]]:
    """
    Recursive character splitter calibrated roughly to token sizes.
    A token is ~4 chars in English, so we multiply.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS * 4,
        chunk_overlap=CHUNK_OVERLAP_TOKENS * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return list(enumerate(splitter.split_text(text)))


def chunk_semantic(text: str, embedder=None) -> list[tuple[int, str]]:
    """
    Semantic chunking via LangChain's SemanticChunker (experimental).
    Splits at semantic similarity drops between consecutive sentences.

    Pass in an OpenAIEmbeddings instance to avoid re-creating it per call.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    if embedder is None:
        embedder = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile")
    chunks = splitter.split_text(text)
    return list(enumerate(chunks))


STRATEGIES = {
    "fixed_size": chunk_fixed_size,
    "semantic": chunk_semantic,
}


def chunk(text: str, strategy: str = "fixed_size") -> list[tuple[int, str]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    return STRATEGIES[strategy](text)

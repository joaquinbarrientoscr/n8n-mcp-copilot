"""
Generation layer: takes a question + retrieved chunks, returns a grounded answer.

Uses OpenAI's chat completion API. Default model: gpt-4o-mini (good cost/quality
for documentation Q&A). For higher quality, set GENERATION_MODEL=gpt-4o.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from generation.prompts import SYSTEM_PROMPT, build_user_prompt
from retrieval.vector_store import RetrievedChunk

load_dotenv()

MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
MAX_TOKENS = 1024

_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


@dataclass
class Answer:
    text: str
    chunks: list[RetrievedChunk]
    model: str


def generate(query: str, chunks: list[RetrievedChunk]) -> Answer:
    client = _client_singleton()
    user_prompt = build_user_prompt(query, chunks)

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = resp.choices[0].message.content or ""
    return Answer(text=text, chunks=chunks, model=MODEL)


# Convenience: full pipeline (retrieve + generate)
def answer(query: str, top_k: int = 5) -> Answer:
    from retrieval.retriever import retrieve
    chunks = retrieve(query, top_k=top_k)
    return generate(query, chunks)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How do I create a webhook trigger in n8n?"
    a = answer(q)
    print(a.text)
    print("\n— Sources —")
    for c in a.chunks:
        print(f"  {c.source_url}  (similarity={c.similarity:.3f})")

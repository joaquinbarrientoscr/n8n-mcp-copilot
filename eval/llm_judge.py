"""
LLM-as-judge scoring for generated answers, using OpenAI.

Uses OpenAI's native JSON mode (response_format={"type": "json_object"}) so
the parse is robust without regex.

Bias warning: judging gpt-4o-mini output with gpt-4o-mini has same-model bias.
Acceptable for v1; in production, switch the judge to a different model family.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM = """You are a strict evaluator of question-answering systems. \
You score answers on two axes, each 1–5:

- faithfulness: 5 = every claim is supported by the provided context;
                1 = the answer fabricates or contradicts the context.
- relevance:    5 = directly answers the question;
                1 = ignores or sidesteps the question.

Output ONLY a JSON object with keys: faithfulness (int), relevance (int),
justification (string, 30 words or less). Nothing else.
"""


@dataclass
class JudgeVerdict:
    faithfulness: int
    relevance: int
    justification: str


_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def judge(question: str, answer: str, context_chunks: list[str]) -> JudgeVerdict:
    client = _client_singleton()
    context = "\n\n---\n\n".join(context_chunks)
    user = (
        f"# Context\n\n{context}\n\n"
        f"# Question\n\n{question}\n\n"
        f"# Answer to evaluate\n\n{answer}\n\n"
        f"Return the JSON verdict now."
    )

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(text)
        return JudgeVerdict(
            faithfulness=int(data["faithfulness"]),
            relevance=int(data["relevance"]),
            justification=str(data.get("justification", "")),
        )
    except Exception as e:                                # noqa: BLE001
        return JudgeVerdict(
            faithfulness=0, relevance=0,
            justification=f"Parse error: {e!s} | raw: {text[:200]}",
        )

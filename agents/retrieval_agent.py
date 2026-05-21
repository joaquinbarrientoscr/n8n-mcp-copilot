"""
Agentic retrieval flow using LangGraph.

Multi-node graph:

    START
      │
      ▼
   [classify_query]  ── conditional edge ──┐
      │                                    │
   ┌──┴──────┐                              │
   ▼         ▼                              ▼
 refuse   clarify                       retrieve
   │         │                              │
   │         │                              ▼
   │         │                          generate
   │         │                              │
   └─────────┴──────────────┬───────────────┘
                            ▼
                           END
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

from generation.generator import generate as rag_generate
from retrieval.retriever import retrieve as rag_retrieve

load_dotenv()

CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini")


class CopilotState(TypedDict, total=False):
    query: str
    classification: str
    classification_reason: str
    chunks: list
    answer: str


_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


CLASSIFIER_SYSTEM = """You are a query classifier for a documentation assistant that \
covers ONLY n8n (workflow automation) and the Model Context Protocol (MCP).

Classify the user's query into exactly one of:
- "in_scope": clear, specific question about n8n or MCP that can be answered from documentation.
- "out_of_scope": question about something else (e.g., AWS, Django, general programming).
- "needs_clarification": about n8n or MCP but too vague to answer well (e.g., "tell me about n8n").

Output ONLY a JSON object: {"classification": "...", "reason": "<one short sentence>"}
"""


def classify_query_node(state: CopilotState) -> CopilotState:
    client = _client_singleton()
    resp = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        max_tokens=150,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": state["query"]},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(text)
        classification = data.get("classification", "in_scope")
        reason = data.get("reason", "")
    except Exception:
        classification = "in_scope"
        reason = "classifier parse error; defaulting to in_scope"
    return {**state, "classification": classification, "classification_reason": reason}


def refuse_node(state: CopilotState) -> CopilotState:
    return {
        **state,
        "answer": (
            "That question is outside what I can help with. I only cover n8n "
            "(workflow automation) and the Model Context Protocol (MCP). "
            "If your question is actually about one of those, try rephrasing it."
        ),
        "chunks": [],
    }


def clarify_node(state: CopilotState) -> CopilotState:
    client = _client_singleton()
    prompt = (
        f"The user asked: '{state['query']}'\n\n"
        "This question is about n8n or MCP but too vague. Ask ONE concise clarifying "
        "question (1–2 sentences) to narrow it down. No apologies, no preamble."
    )
    resp = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    return {**state, "answer": text, "chunks": []}


def retrieve_node(state: CopilotState) -> CopilotState:
    chunks = rag_retrieve(state["query"], top_k=5)
    return {**state, "chunks": chunks}


def generate_node(state: CopilotState) -> CopilotState:
    ans = rag_generate(state["query"], state["chunks"])
    return {**state, "answer": ans.text}


def route_after_classify(state: CopilotState) -> str:
    c = state.get("classification", "in_scope")
    if c == "out_of_scope":
        return "refuse"
    if c == "needs_clarification":
        return "clarify"
    return "retrieve"


def build_graph():
    g = StateGraph(CopilotState)
    g.add_node("classify_query", classify_query_node)
    g.add_node("refuse", refuse_node)
    g.add_node("clarify", clarify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("classify_query")
    g.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {"refuse": "refuse", "clarify": "clarify", "retrieve": "retrieve"},
    )
    g.add_edge("refuse", END)
    g.add_edge("clarify", END)
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def answer(query: str) -> CopilotState:
    graph = get_graph()
    return graph.invoke({"query": query})


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How do I create a webhook trigger in n8n?"
    print(f"Query: {q}\n")
    print("─" * 60)
    result = answer(q)
    print(f"Classification: {result.get('classification')}")
    print(f"Reason: {result.get('classification_reason')}")
    print("─" * 60)
    print(f"\nAnswer:\n{result.get('answer')}")
    if result.get("chunks"):
        print("\n— Sources —")
        for c in result["chunks"]:
            print(f"  {c.source_url}  (sim={c.similarity:.3f})")
"""
Streamlit UI for the n8n + MCP Copilot.

Launch:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running via `streamlit run ui/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import streamlit as st
from dotenv import load_dotenv

from generation.generator import generate
from retrieval.retriever import retrieve

load_dotenv()

st.set_page_config(
    page_title=os.getenv("APP_TITLE", "n8n + MCP Copilot"),
    page_icon="🧭",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align:center'>🧭 n8n + MCP Copilot</h1>"
    "<p style='text-align:center;color:#777'>"
    "Ask anything about n8n or the Model Context Protocol. Answers are grounded "
    "in the official documentation."
    "</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider("top_k", 1, 15, int(os.getenv("DEFAULT_TOP_K", "5")))
    strategy = st.selectbox(
        "Chunking strategy",
        options=["fixed_size", "semantic"],
        index=0 if os.getenv("DEFAULT_CHUNKING_STRATEGY", "fixed_size") == "fixed_size" else 1,
    )
    doc_filter = st.selectbox(
        "Filter by source",
        options=["all", "n8n", "mcp"],
        index=0,
    )

query = st.text_input(
    "Question",
    placeholder="e.g. How do I trigger a workflow with a webhook in n8n?",
)

if st.button("Ask", type="primary") and query:
    with st.spinner("Retrieving…"):
        chunks = retrieve(
            query,
            top_k=top_k,
            strategy=strategy,
            doc_type=None if doc_filter == "all" else doc_filter,
        )

    if not chunks:
        st.warning("No relevant context found. Try rephrasing.")
    else:
        with st.spinner("Generating answer…"):
            ans = generate(query, chunks)

        st.markdown("### Answer")
        st.markdown(ans.text)

        st.markdown("### Sources")
        for c in chunks:
            st.markdown(
                f"- **[{c.title or c.source_url}]({c.source_url})** "
                f"<span style='color:#999'>(similarity {c.similarity:.3f})</span>",
                unsafe_allow_html=True,
            )

        with st.expander("Retrieved chunks (debug)"):
            for i, c in enumerate(chunks, start=1):
                st.markdown(
                    f"**{i}.** [{c.doc_type}] {c.source_url} — sim {c.similarity:.3f}"
                )
                st.code(c.content[:800] + ("…" if len(c.content) > 800 else ""))
"""
System prompts for the RAG generator.

Citation enforcement and refusal rules are the two strongest non-eval guardrails
against hallucination — see DECISIONS.md (D-006).
"""

SYSTEM_PROMPT = """You are an expert assistant for engineers building automations \
with n8n and the Model Context Protocol (MCP).

You answer questions based ONLY on the documentation context provided below. \
You always follow these rules:

1. **Cite every claim.** After each factual statement, include the source URL \
   in brackets: [source: <url>]. Use the URLs of the chunks provided in context.

2. **Refuse to guess.** If the context does not contain the answer, say \
   "I don't have that in the documentation I've been given — you may want to \
   check the official docs directly." Never fabricate APIs, function names, \
   or behavior.

3. **Be concrete.** Prefer code examples, configuration snippets, and step lists \
   over abstract description, when the context supports it.

4. **Distinguish n8n from MCP.** They are different systems. Don't conflate \
   concepts unless the user explicitly asks how they integrate.

5. **Be brief.** Use the minimum words to answer correctly. No "Great question!" \
   preambles, no recap of the question.
"""


def build_user_prompt(query: str, chunks: list) -> str:
    """
    Build the user message that includes retrieved context plus the question.
    `chunks` is a list of RetrievedChunk.
    """
    if not chunks:
        return f"Question: {query}\n\n(No context was retrieved.)"

    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"---\nSource: {c.source_url}\nTitle: {c.title}\n\n{c.content}"
        )
    context = "\n\n".join(context_blocks)

    return (
        f"# Documentation context\n\n{context}\n\n"
        f"# Question\n\n{query}\n\n"
        f"Answer the question using only the documentation context above. "
        f"Cite the source URL after each claim."
    )

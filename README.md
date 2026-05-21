# n8n + MCP Copilot

A retrieval-augmented AI assistant for engineers building automations with
[n8n](https://n8n.io) and the [Model Context Protocol](https://modelcontextprotocol.io).

Grounds every answer in the official documentation, cites its sources, and
keeps an evaluation harness honest about retrieval quality.

> **Built as a focused weekend project to ship a production-shape RAG system
> end-to-end: ingestion → embedding → retrieval → generation → evaluation,
> with documented trade-offs in `DECISIONS.md`.**

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Docs sources │ ──▶ │  Ingestion   │ ──▶ │  Postgres +      │
│ (n8n, MCP)   │     │ scrape→chunk │     │  pgvector        │
└──────────────┘     │ → embed      │     │  (1536-dim)      │
                     └──────────────┘     └────────┬─────────┘
                                                    │
                          ┌─────────────────────────┘
                          ▼
                   ┌─────────────┐     ┌──────────────┐     ┌────────────┐
                   │  Retriever  │ ──▶ │  Generator   │ ──▶ │ Streamlit  │
                   │ cosine top-k│     │ Claude + ctx │     │    UI      │
                   │ + rerank    │     │ + citations  │     │            │
                   └─────────────┘     └──────────────┘     └────────────┘

                                                                  ▲
                          ┌──────────────────────┐                │
                          │ Evaluation Harness   │ ───────────────┘
                          │ precision@k, MRR,    │
                          │ LLM-as-judge         │
                          └──────────────────────┘
```

## Stack

| Layer        | Choice                                  | Why                                      |
|--------------|-----------------------------------------|------------------------------------------|
| Vector store | PostgreSQL + pgvector                   | Self-hosted, no ext. service, enterprise |
| Embeddings   | OpenAI `text-embedding-3-small` (1536d) | $0.02/1M tokens, strong quality          |
| Generation   | Anthropic Claude (Sonnet)               | Reasoning quality on technical Q&A       |
| Orchestration| LangChain + LangGraph (agentic mode)    | Industry-standard, swappable             |
| UI           | Streamlit                               | Fast to ship, good for demos             |
| Eval         | Custom harness + LLM-as-judge           | Reproducible numbers                     |

See `DECISIONS.md` for trade-offs (chunking strategy, top-k, rerank, etc.).

## Quick start

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- OpenAI API key (for embeddings)
- Anthropic API key (for generation)

### Setup

```bash
# 1. Clone & enter
git clone <your-repo> n8n-mcp-copilot && cd n8n-mcp-copilot

# 2. Python env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Env vars
cp .env.example .env
# edit .env: paste your OPENAI_API_KEY and ANTHROPIC_API_KEY

# 4. Start Postgres + pgvector
docker compose up -d

# 5. Ingest the docs (5–15 min depending on MAX_PAGES_PER_SOURCE)
python -m ingestion.ingest

# 6. Launch UI
streamlit run ui/app.py
```

Open <http://localhost:8501>.

### Evaluation

```bash
python -m eval.run_eval
```

Reports `precision@k`, `recall@k`, `MRR` for retrieval and a
faithfulness/relevance score (LLM-as-judge) for generation.

---

## Weekend execution plan

This repo is scaffolded for a focused 2-day build. Use it as the spine.

### Saturday (~8h)
- [ ] **Hour 1–2**: Setup. Read every file once. Configure `.env`. `docker compose up`.
- [ ] **Hour 3–5**: Run `ingestion.ingest`. Verify docs and chunks land in DB.
      Inspect chunks manually (`psql` or DBeaver) and sanity-check quality.
- [ ] **Hour 6–7**: Run a few retrieval queries via `retrieval/retriever.py`
      directly. Verify top-k results make sense.
- [ ] **Hour 8**: Bring up Streamlit. Test end-to-end with 5 real queries.
      Take screenshots — they go in the README later.
- [ ] **Hour 9–10**: Write 30 query/answer pairs in `data/eval/eval_set.jsonl`.

### Sunday (~7h)
- [ ] **Hour 1–2**: Run `eval/run_eval.py`. Record baseline numbers in `DECISIONS.md`.
- [ ] **Hour 3–4**: Experiment. Try `chunking_strategy=semantic` vs `fixed_size`,
      different `top_k` values. Re-run eval. Document winners.
- [ ] **Hour 5**: Implement `retrieval/rerank.py` (Cohere or cross-encoder).
      Compare with/without.
- [ ] **Hour 6** *(bonus)*: Wire up `agents/retrieval_agent.py` (LangGraph) —
      clarification node → retrieval node → generation node.
- [ ] **Hour 7**: Polish README with screenshots, real numbers, "what I'd do next".
      Deploy to VPS. Push to GitHub.

---

## Project layout

```
n8n-mcp-copilot/
├── docker-compose.yml          # Postgres + pgvector
├── db/init.sql                 # Schema (documents, chunks tables)
├── ingestion/
│   ├── scrape.py               # Pull docs from n8n + MCP
│   ├── chunk.py                # Fixed-size and semantic chunking
│   ├── embed.py                # OpenAI embeddings
│   └── ingest.py               # End-to-end orchestrator
├── retrieval/
│   ├── vector_store.py         # pgvector interface
│   ├── retriever.py            # Top-k similarity search
│   └── rerank.py               # Reranker (day 2)
├── generation/
│   ├── prompts.py              # System prompts with citation rules
│   └── generator.py            # Claude generation with context
├── agents/
│   └── retrieval_agent.py      # LangGraph agentic flow (bonus)
├── eval/
│   ├── metrics.py              # precision@k, recall@k, MRR
│   ├── llm_judge.py            # Faithfulness/relevance scoring
│   └── run_eval.py             # Eval runner
├── data/eval/eval_set.jsonl    # Ground truth (30 pairs target)
├── ui/app.py                   # Streamlit interface
└── DECISIONS.md                # Trade-offs documented in flight
```

---

## License

MIT — built by [Joaquín Barrientos](https://joaquinbarrientoscr.com).

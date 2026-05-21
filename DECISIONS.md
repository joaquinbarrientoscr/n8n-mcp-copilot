# Engineering Decisions

> Each decision is documented as it's made, with the alternatives considered
> and the trade-off that drove the choice. Fill these in as you build — this
> document is what makes the project defensible in technical interviews.

---

## D-001: Vector store — Postgres + pgvector

**Decision**: PostgreSQL with the `pgvector` extension.

**Alternatives considered**:
- **Pinecone** — managed, fast, but external dependency and pricing tier per project.
- **Chroma** — easy local setup, but less production-grade indexing options.
- **Weaviate** — feature-rich, but heavier ops overhead.
- **FAISS (flat file)** — fastest local, but no SQL, no metadata filtering.

**Trade-off**: Picked pgvector for self-hosted control, no monthly cost, native
SQL filtering on metadata, and because every team running enterprise software
already operates Postgres — minimal new infra to learn.

**Cost of being wrong**: At <1M chunks, pgvector with IVFFlat is more than fast
enough; if scale demands sub-100ms p95 over 10M+ chunks, migration to a
dedicated vector DB is a known path.

---

## D-002: Embedding model — OpenAI `text-embedding-3-small`

**Decision**: OpenAI's `text-embedding-3-small`, 1536 dimensions.

**Alternatives considered**:
- **`text-embedding-3-large` (3072d)** — higher quality, ~6× cost.
- **Voyage `voyage-3`** — strong on code/technical text but extra API key.
- **`sentence-transformers/all-MiniLM-L6-v2` (local, 384d)** — free, but lower quality.

**Trade-off**: `text-embedding-3-small` hits the cost-quality sweet spot for a
mid-size technical corpus ($0.02/1M tokens). Easy to swap later via `.env` —
the schema stores `EMBEDDING_DIMENSION` and the code reads it.

**To verify**: Re-run eval with `voyage-3` once and compare MRR. (Day 2 stretch.)

---

## D-003: Chunking strategy — TBD after experiment

**Decision** *(to record on Saturday afternoon)*: ___

**Alternatives**:
- **Fixed-size (512 tokens, 50 overlap)** — simple, predictable.
- **Semantic chunking** (LangChain `SemanticChunker`) — splits at semantic
  boundaries; usually higher retrieval quality but slower to ingest.
- **Markdown-aware** — split on `##` headers; great for docs but brittle if
  source HTML→MD conversion is messy.

**Method**: Ingest the corpus twice (once per strategy, distinguished by the
`chunking_strategy` column) and run the eval harness against each.

**Result** *(fill in)*: precision@5 fixed = ___ , semantic = ___ ; chose ___.

---

## D-004: Top-k retrieval — TBD after experiment

**Decision** *(record after sweep)*: ___

**Sweep**: Run eval with `top_k ∈ {3, 5, 10}`. Record precision@k and recall@k.
Pick the k that maximises retrieval quality without bloating the LLM context.

**Result** *(fill in)*: ___

---

## D-005: Reranking — TBD

**Decision** *(record on Sunday afternoon)*: ___

**Approach**: Two-stage retrieval — pull top-20 by cosine, then rerank with a
cross-encoder or Cohere `rerank-3`, keep top-5.

**Result** *(fill in)*: Δ precision@5 with/without reranking = ___

---

## D-006: Generation prompt — citation enforcement

**Decision**: The system prompt requires the model to cite source URLs inline
for every claim, and to say "I don't know" rather than guess if retrieval
returned nothing relevant.

**Why**: Hallucination is the dominant failure mode for documentation RAG.
Citation requirements + a strong refusal rule are the simplest non-eval
guardrails.

**Limit**: Citation accuracy is not (yet) automatically verified — only
faithfulness via LLM-as-judge. A future pass could parse citations and
validate they match retrieved chunks.

---

## D-007: Evaluation — LLM-as-judge for answer quality

**Decision**: Use Claude (judge model = generation model is fine for v1) to
score faithfulness (grounded in context?) and relevance (answers the question?)
on a 1–5 scale.

**Risk**: Same-model judge bias. For a public-facing product, would switch
the judge to a different model family (e.g., GPT-4o judging Claude output).

---

## D-008: Stack omissions (intentional)

- **No hybrid search (BM25 + vector)**: Would likely improve precision on
  literal keyword matches. Skipped for time.
- **No query rewriting**: Single-turn retrieval only. Multi-turn refinement
  is a natural day-3 addition.
- **No streaming**: UI returns full response; streaming token-by-token is a
  UX nicety, not a quality lift.
- **No auth/rate-limiting on the demo**: This is a portfolio project, not
  multi-tenant production. Documented as a "next steps" item.

## D-009: Pipeline de dos fases para resiliencia

**Problem discovered**: The current ingestion pipeline does scrape → chunk → embed
in a single loop, persisting to DB only when a flush batches 500 chunks. On the
first ingestion run a failure in the embed step (invalid OPENAI_API_KEY placeholder)
aborted the entire pipeline mid-way, losing all in-memory scraped content from
both n8n (150 pages) and MCP (10 pages partial). Re-running required re-scraping
from zero.

**Proposed redesign**: Split the pipeline into two independent phases:
1. **Scrape-and-persist**: scrape pages and immediately upsert into the
   `documents` table with `raw_content`. No chunking or embedding in this phase.
2. **Chunk-and-embed**: read documents from DB that don't yet have chunks for
   the target `chunking_strategy`, process them, write chunks back. Idempotent
   and resumable.

**Why it matters**: scraping is the most expensive failure to recover from
(network I/O, rate-limited, slow). Embedding failures are cheaper but more
common (API key issues, rate limits, transient errors). Decoupling means a
failure in the cheap phase doesn't trash the expensive phase.

**Alternatives considered**:
- *Checkpoint scraping to disk* (JSONL per page before embedding): works, but
  adds a parallel storage layer to manage. Postgres-only is cleaner.
- *Retry embedding with backoff*: handles transient errors but doesn't help
  with hard failures like wrong API keys or schema mismatches.

**Status**: Identified, not yet implemented. Candidate for Layer 2 ownership pass.

**Risk if not addressed**: Any future ingestion failure mid-embedding wastes
the full scraping time (3–15 min depending on MAX_PAGES). Low impact for
one-off runs, higher impact when iterating on chunking strategies — each
strategy re-runs the full pipeline today, including re-scraping the corpus
that didn't change.

## D-010: Corpus coverage — prioritized path ingestion

**Problem discovered**: Initial ingestion with MAX_PAGES_PER_SOURCE=150 took the
first 150 URLs from the n8n sitemap in default order. Eval revealed clear in-scope
queries (e.g., "how do I create a webhook trigger") failed with refusal because
node-specific documentation (under /integrations/builtin/core-nodes/) was never
ingested.

**Decision**: Added _prioritize_urls() to score each URL by path pattern before
applying MAX_PAGES truncation. High-priority patterns: [list yours]. Increased
MAX_PAGES to 400 to ensure full coverage of high+medium priority paths.

**Result**: similarity@1 on "create webhook trigger" went from <pre> to <post>.
[Anotás los números reales.]

**Trade-off**: more pages = more embedding cost (~$0.05 vs $0.015 before) and
longer ingest time (~12 min vs ~3). For a 1M+ page corpus, this approach wouldn't
scale — would need indexed retrieval over a documentation graph instead.

## D-011: Multi-tenant knowledge bases

**Decision**: Refactored from a hard-coded `doc_type` field to a first-class
`knowledge_bases` table, with `documents.knowledge_base_id` as FK.

**Why**: Original design assumed 2 static corpora (n8n, MCP). Real enterprise
deployment requires multiple isolated knowledge bases (HR, Operations, IT,
Compliance) and the ability to add new ones without schema migrations.

**Architectural choice**: Static caller-side filtering (the UI/agent caller
declares which KBs to search). Alternative considered: LLM-based KB routing
where the classifier infers the right KB from the query. Rejected for v1
because (a) the explicit choice is more predictable for users, (b) wrong KB
routing fails silently in ways that are hard to debug, (c) explicit is easier
to migrate to RBAC in the future.

**Generalization demonstrated**: Added a third KB (Streamlit docs / whatever)
via `ingestion/add_kb.py` to show the pattern actually generalizes beyond
the original two.

## D-012: HTML text extraction quality — root cause of ranking failures

**Problem discovered**: During eval, "how do I create a webhook trigger in n8n"
returned content from Facebook integration docs despite the dedicated Webhook
node page being in the corpus. Top-20 retrieval did not include the correct
page at all.

**Initial hypotheses** (all wrong): coverage gap, chunking granularity,
conceptual-vs-operational content mismatch.

**Actual root cause**: BeautifulSoup `get_text(separator="\n")` was inserting
line breaks between inline HTML elements (code, a, strong, em). Natural
sentences like "Use the Webhook node to create webhooks" were fragmenting into
"Use the Webhook node to create / webhooks / ,". Embedding quality of the
resulting text was degraded enough that the conceptual Webhook page never
ranked in top-20.

**Fix**: Unwrap inline elements before extraction:
    for inline_tag in main.find_all(['code', 'a', 'strong', 'em', 'span', 'b', 'i']):
        inline_tag.unwrap()
    text = main.get_text(separator="\n", strip=True)

**Impact**: [fill in before/after numbers — top-1 similarity, target page rank]

**Lesson**: In RAG, text extraction quality is upstream of everything else.
No amount of reranking, hybrid search, or query rewriting fixes a corpus
where natural sentence flow is broken at ingest time.

**Impact**:
- Before: top-1 = facebookapp/ credentials (sim=0.596), webhook node page absent from top-20
- After: top-1 = webhook-url configuration page (sim=0.564), entire top-10 is webhook/n8n-config related
- Root cause confirmed: BeautifulSoup NavigableString fragmentation after unwrap().
  Fix: main.smooth() consolidates adjacent strings before get_text().

## D-013: Query semantic mismatch — conversational vs documentation vocabulary

**Problem**: Conversational queries like "how do I create a webhook trigger"
return hosting/configuration pages instead of the Webhook node documentation.
Both contain high density of "create", "configure", "webhook", "trigger" but
in different contexts. The node-doc page ranks below config pages in cosine
similarity despite being more relevant to the intent.

**Root cause**: Embedding models encode semantic proximity, not user intent.
Hosting configuration pages have operationally dense text that overlaps
with the query vocabulary at higher similarity than conceptual node docs.

**Proposed solution**: Add a query rewriting node to the LangGraph agent
(classify → rewrite → retrieve → generate). The rewriter transforms
conversational phrasing into documentation-style terms before retrieval.
e.g. "how do I create a webhook trigger in n8n" → "n8n Webhook node usage"

**Status**: Identified, not yet implemented. Clear path forward.

---

## What I'd do next, with more time

1. **Hybrid retrieval** (vector + BM25) for better keyword-match queries.
2. **Query rewriting** — use the LLM to expand or clarify before retrieval.
3. **Citation validation** — parse the generator output and verify every cited
   URL was in the retrieved chunks.
4. **Multi-tenant isolation** — Session ID per user, RLS at the chunk level
   if serving multiple clients.
5. **Observability** — log every retrieval (query, top-k IDs, scores, judge
   verdict) for offline analysis.
6. **Embedding model sweep** at scale — `voyage-3`, `text-embedding-3-large`.

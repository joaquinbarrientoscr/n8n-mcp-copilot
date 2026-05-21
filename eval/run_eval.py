"""
Run the evaluation harness against data/eval/eval_set.jsonl.

For each query:
  1. Retrieve top-k chunks.
  2. Compute retrieval metrics (precision@k, recall@k, MRR) against relevant_urls.
  3. Generate the answer.
  4. Judge the answer (faithfulness, relevance) with LLM-as-judge.

Aggregate across the whole set and print a summary table. Save per-query
results to eval/runs/<timestamp>.jsonl for inspection.

Usage:
    python -m eval.run_eval                              # default top_k, strategy
    python -m eval.run_eval --strategy semantic --top_k 10
    python -m eval.run_eval --skip-judge                 # retrieval-only (faster)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from eval.llm_judge import judge
from eval.metrics import aggregate, precision_at_k, recall_at_k, reciprocal_rank
from generation.generator import generate
from retrieval.retriever import retrieve

load_dotenv()

EVAL_PATH = Path("data/eval/eval_set.jsonl")
RUNS_DIR = Path("eval/runs")


def load_eval_set(path: Path = EVAL_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Eval set not found at {path}. "
            f"Write ~30 query/answer pairs there in JSONL format. "
            f"See data/eval/eval_set.jsonl for the schema."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(strategy: str, top_k: int, skip_judge: bool) -> None:
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval queries.")
    print(f"Strategy: {strategy} | top_k: {top_k} | judge: {not skip_judge}")
    print("─" * 60)

    per_query = []

    for item in tqdm(eval_set, desc="Evaluating"):
        query = item["query"]
        relevant = set(item.get("relevant_urls", []))

        chunks = retrieve(query, top_k=top_k, strategy=strategy)
        retrieved_urls = [c.source_url for c in chunks]

        row = {
            "query": query,
            "precision@k": precision_at_k(retrieved_urls, relevant, top_k),
            "recall@k": recall_at_k(retrieved_urls, relevant, top_k),
            "mrr": reciprocal_rank(retrieved_urls, relevant),
            "retrieved_urls": retrieved_urls,
        }

        if not skip_judge:
            ans = generate(query, chunks)
            verdict = judge(query, ans.text, [c.content for c in chunks])
            row["answer"] = ans.text
            row["faithfulness"] = verdict.faithfulness
            row["relevance"] = verdict.relevance
            row["judge_note"] = verdict.justification

        per_query.append(row)

    summary = aggregate(per_query)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    for k, v in summary.items():
        print(f"  {k:15s} {v}")
    print("═" * 60)

    # Persist run
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{int(time.time())}_{strategy}_k{top_k}.jsonl"
    with out_path.open("w") as f:
        for row in per_query:
            f.write(json.dumps(row) + "\n")
    print(f"\n✓ Per-query results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=os.getenv("DEFAULT_CHUNKING_STRATEGY", "fixed_size"))
    parser.add_argument("--top_k", type=int, default=int(os.getenv("DEFAULT_TOP_K", "5")))
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-as-judge (retrieval metrics only)")
    args = parser.parse_args()
    run(strategy=args.strategy, top_k=args.top_k, skip_judge=args.skip_judge)

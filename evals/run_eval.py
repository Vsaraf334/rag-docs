"""Run the RAG system over evals/questions.jsonl and report accuracy.

Each line of questions.jsonl looks like:
{"question": "...", "source": "file.md" or null, "expected": "a phrase the answer must contain"}

Run:  uv run --env-file .env python evals/run_eval.py [--backend chroma|pg|hybrid] [--top-k 8]
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from judge import judge

from answer import generate
from pg_store import retrieve_hybrid_pg, retrieve_pg
from retrieve import retrieve

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RETRIEVERS: dict[str, Callable[..., list[dict]]] = {
    "chroma": retrieve,
    "pg": retrieve_pg,
    "hybrid": retrieve_hybrid_pg,
}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(q: dict, retriever: Callable[..., list[dict]], top_k: int) -> dict:
    """Run one question through retrieval and generation, and score both."""
    chunks = retriever(q["question"], top_k=top_k)
    sources = [c["source"] for c in chunks]
    answer = generate(q["question"], chunks)
    result = judge(q["question"], q["expected"], q["source"] is not None, answer)
    return {
        **q,
        "retrieved": sources,
        "hit": q["source"] in sources,
        "answer": answer,
        "verdict": result.verdict,
        "reason": result.reason,
    }


def main(backend: str, top_k: int) -> None:
    print(f"backend = {backend}, top_k = {top_k}\n")
    retriever = RETRIEVERS[backend]
    results = [evaluate(q, retriever, top_k) for q in load_questions()]
    answerable = [r for r in results if r["source"] is not None]
    unanswerable = [r for r in results if r["source"] is None]

    for r in results:
        if r["source"] is not None and not r["hit"]:
            print(f"RETRIEVAL MISS  {r['question']}")
            print(f"    expected {r['source']}, got {r['retrieved']}")
        if r["verdict"] != "correct":
            print(f"ANSWER {r['verdict'].upper():<9}{r['question']}")
            print(f"    {r['reason']}")

    hits = sum(r["hit"] for r in answerable)
    correct = sum(r["verdict"] == "correct" for r in answerable)
    partial = sum(r["verdict"] == "partial" for r in answerable)
    refused = sum(r["verdict"] == "correct" for r in unanswerable)
    print()
    print(f"Retrieval hit rate:   {hits}/{len(answerable)}")
    print(f"Answer accuracy:      {correct}/{len(answerable)}  ({partial} partial)")
    print(f"Correct refusals:     {refused}/{len(unanswerable)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--backend", choices=RETRIEVERS, default="chroma")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    main(args.backend, args.top_k)

"""Run the RAG system over evals/questions.jsonl and report accuracy.

Each line of questions.jsonl looks like:
{"question": "...", "source": "file.md" or null, "expected": "a phrase the answer must contain"}

Run:  uv run --env-file .env python evals/run_eval.py [--mode pipeline|agent]
                                                      [--backend chroma|pg] [--top-k 8]
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from judge import judge

from agent import agent_answer
from answer import generate_message
from pg_store import retrieve_pg
from retrieve import retrieve

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RETRIEVERS: dict[str, Callable[..., list[dict]]] = {
    "chroma": retrieve,
    "pg": retrieve_pg,
}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(
    question: str, retriever: Callable[..., list[dict]], top_k: int
) -> dict:
    """Fixed workflow: one search with the user's words, one API call."""
    chunks = retriever(question, top_k=top_k)
    response = generate_message(question, chunks)
    return {
        "answer": "".join(b.text for b in response.content if b.type == "text"),
        "sources": [c["source"] for c in chunks],
        "searches": [question],
        "calls": 1,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def run_agent(question: str, retriever: Callable[..., list[dict]], top_k: int) -> dict:
    """Agent: Claude decides what to search for, and how many times."""
    try:
        return agent_answer(question, retriever, top_k)
    except RuntimeError as e:  # stopped early (max_tokens, refusal, out of steps)
        return {
            "answer": f"ERROR: {e}",
            "sources": [],
            "searches": [],
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


MODES = {"pipeline": run_pipeline, "agent": run_agent}


def evaluate(
    q: dict, mode: str, retriever: Callable[..., list[dict]], top_k: int
) -> dict:
    """Run one question through the chosen mode, then score retrieval and answer."""
    start = time.perf_counter()
    run = MODES[mode](q["question"], retriever, top_k)
    seconds = time.perf_counter() - start
    result = judge(q["question"], q["expected"], q["source"] is not None, run["answer"])
    return {
        **q,
        **run,
        "seconds": seconds,
        "hit": q["source"] in run["sources"],
        "verdict": result.verdict,
        "reason": result.reason,
    }


def print_cost(results: list[dict]) -> None:
    """Averages and worst cases: agents trade cost and latency for quality."""
    n = len(results)
    for key, label in [
        ("calls", "API calls"),
        ("searches", "Searches"),
        ("input_tokens", "Input tokens"),
        ("seconds", "Seconds"),
    ]:
        values = [len(r[key]) if key == "searches" else r[key] for r in results]
        print(
            f"{label + ' per question:':<26}avg {sum(values) / n:>8.1f}   max {max(values):>8.1f}"
        )
    # Haiku 5.5 prices: $0.10 / $0.50 per million input / output tokens (judge not included)
    cost = (
        sum(r["input_tokens"] * 0.10 + r["output_tokens"] * 0.50 for r in results) / 1e6
    )
    print(f"{'Answer cost, all questions:':<26}${cost:.4f}")


def main(mode: str, backend: str, top_k: int) -> None:
    print(f"mode = {mode}, backend = {backend}, top_k = {top_k}\n")
    retriever = RETRIEVERS[backend]
    results = [evaluate(q, mode, retriever, top_k) for q in load_questions()]
    answerable = [r for r in results if r["source"] is not None]
    unanswerable = [r for r in results if r["source"] is None]

    for r in results:
        if r["source"] is not None and not r["hit"]:
            print(f"RETRIEVAL MISS  {r['question']}")
            print(f"    expected {r['source']}, got {sorted(set(r['sources']))}")
            if mode == "agent":
                print(f"    searched {r['searches']}")
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
    print()
    print_cost(results)
    print_cost_split(answerable, unanswerable)

    out = QUESTIONS.parent / f"results-{mode}-{backend}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)
    print(f"\nDetails written to {out.name}")


def print_cost_split(answerable: list[dict], unanswerable: list[dict]) -> None:
    """Average input tokens for answerable vs unanswerable questions."""
    for name, group in [("answerable", answerable), ("unanswerable", unanswerable)]:
        avg = sum(r["input_tokens"] for r in group) / len(group)
        print(f"{'Avg tokens, ' + name + ':':<26}{avg:>12.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG system.")
    parser.add_argument("--mode", choices=MODES, default="pipeline")
    parser.add_argument("--backend", choices=RETRIEVERS, default="chroma")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    main(args.mode, args.backend, args.top_k)

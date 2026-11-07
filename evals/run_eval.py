"""Run the RAG system over evals/questions.jsonl and report accuracy.

Each line of questions.jsonl looks like:
{"question": "...", "source": "file.md" or null, "expected": "a phrase the answer must contain"}

Run:  uv run --env-file .env python evals/run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from judge import judge

from answer import generate
from retrieve import retrieve

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(q: dict) -> dict:
    """Run one question through retrieval and generation, and score both."""
    chunks = retrieve(q["question"])
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


def main() -> None:
    results = [evaluate(q) for q in load_questions()]
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
    main()

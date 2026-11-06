"""Run the RAG system over evals/questions.jsonl and report accuracy.

Each line of questions.jsonl looks like:
{"question": "...", "source": "file.md" or null, "expected": "a phrase the answer must contain"}

Run:  uv run --env-file .env python evals/run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from answer import generate
from retrieve import retrieve

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_correct(answer: str, expected: str) -> bool:
    # Simple containment check to start. Replace with an LLM judge later if needed.
    normalized = answer.lower().replace("’", "'")  # curly apostrophe -> straight
    return expected.lower() in normalized


def evaluate(q: dict) -> dict:
    """Run one question through retrieval and generation, and score both."""
    chunks = retrieve(q["question"])
    sources = [c["source"] for c in chunks]
    answer = generate(q["question"], chunks)
    return {
        **q,
        "retrieved": sources,
        "hit": q["source"] in sources,
        "answer": answer,
        "correct": is_correct(answer, q["expected"]),
    }


def main() -> None:
    results = [evaluate(q) for q in load_questions()]
    answerable = [r for r in results if r["source"] is not None]
    unanswerable = [r for r in results if r["source"] is None]

    for r in results:
        if r["source"] is not None and not r["hit"]:
            print(f"RETRIEVAL MISS  {r['question']}")
            print(f"    expected {r['source']}, got {r['retrieved']}")
        if not r["correct"]:
            print(f"ANSWER MISS     {r['question']}")
            print(f"    expected '{r['expected']}' in: {r['answer'][:200]!r}")

    hits = sum(r["hit"] for r in answerable)
    correct = sum(r["correct"] for r in answerable)
    refused = sum(r["correct"] for r in unanswerable)
    print()
    print(f"Retrieval hit rate:   {hits}/{len(answerable)}")
    print(f"Answer accuracy:      {correct}/{len(answerable)}")
    print(f"Correct refusals:     {refused}/{len(unanswerable)}")


if __name__ == "__main__":
    main()

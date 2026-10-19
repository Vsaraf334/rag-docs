"""Run the RAG system over evals/questions.jsonl and report accuracy.

Each line of questions.jsonl looks like:
{"question": "...", "expected": "a short phrase the answer must contain"}
"""

import json
from pathlib import Path

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_correct(answer: str, expected: str) -> bool:
    # Simple containment check to start. Replace with an LLM judge later if needed.
    return expected.lower() in answer.lower()


def main() -> None:
    questions = load_questions()
    # Week 5: replace this with a call to your answer pipeline.
    results = [(q, is_correct("", q["expected"])) for q in questions]
    passed = sum(ok for _, ok in results)
    print(f"Accuracy: {passed}/{len(results)}")


if __name__ == "__main__":
    main()

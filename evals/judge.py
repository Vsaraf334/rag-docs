"""Grade an answer with Claude instead of a substring check."""

from typing import Literal

import anthropic
from pydantic import BaseModel

JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_PROMPT = """You grade answers from a documentation Q&A system.

<question>{question}</question>
<key_fact>{expected}</key_fact>
<answerable>{answerable}</answerable>
<answer>{answer}</answer>

If answerable is true, the answer should contain the key fact (wording may differ):
- "correct": answers the question and agrees with the key fact
- "partial": only partly answers it, or says it doesn't know
- "incorrect": wrong or contradicts the key fact
If answerable is false, the docs don't cover this question:
- "correct": the answer says it doesn't know
- "incorrect": the answer gives a solution anyway

Reply with only JSON: {{"verdict": "...", "reason": "one sentence"}}"""

client = anthropic.Anthropic()


class Verdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str


def judge(question: str, expected: str, answerable: bool, answer: str) -> Verdict:
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, answerable=answerable, answer=answer
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Judge reply was cut off; increase max_tokens")
    text = "".join(b.text for b in response.content if b.type == "text")
    return Verdict.model_validate_json(text)

"""Answer a question from retrieved chunks, citing sources."""

import sys

import anthropic

from retrieve import retrieve

MODEL = "claude-haiku-5-5"
SYSTEM_PROMPT = (
    "You answer questions about FastAPI using only the provided documents. "
    "After each claim, cite its source in square brackets, like [body.md]. "
    "If the documents don't contain the answer, say you don't know. "
    "Never use outside knowledge."
)

client = anthropic.Anthropic()


def format_documents(chunks: list[dict]) -> str:
    """Wrap each chunk in <document> tags. Shared by the pipeline and the agent."""
    return "\n\n".join(
        f'<document source="{c["source"]}">\n{c["text"]}\n</document>' for c in chunks
    )


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Wrap the chunks in <documents>, then add the question."""
    return (
        f"<documents>\n{format_documents(chunks)}\n</documents>\n\nQuestion: {question}"
    )


def answer(question: str) -> str:
    return generate(question, retrieve(question))


def generate_message(question: str, chunks: list[dict]) -> anthropic.types.Message:
    """Ask Claude to answer from the given chunks; return the full response (with usage)."""
    return client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )


def generate(question: str, chunks: list[dict]) -> str:
    """Ask Claude to answer the question using only the given chunks."""
    response = generate_message(question, chunks)
    return "".join(b.text for b in response.content if b.type == "text")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    print(answer(question))

"""Agentic RAG: Claude decides what to search for, and how many times.

The pipeline (answer.py) always searches once, with the user's exact words.
Here search is a tool, so Claude can rephrase the query, search again, or not
search at all. evals/run_eval.py compares the two on the same questions.

Run:  uv run --env-file .env python src/agent.py "How do I return a 404 error?"
"""

import sys
from collections.abc import Callable

import anthropic
from anthropic import beta_tool

from answer import MODEL, format_documents
from retrieve import retrieve

MAX_SEARCHES = 2  # per question; more searches = more chunks re-sent on every call
MAX_STEPS = 6  # API calls per question; the tool runner has no limit by default
SYSTEM_PROMPT = (
    "You answer questions about FastAPI using only the documents returned by the "
    "search_docs tool. Always search before answering. If the results don't answer "
    "the question, search again with different words. "
    "After each claim, cite its source in square brackets, like [body.md]. "
    "If the documents don't contain the answer, say you don't know. "
    "Never use outside knowledge."
)

client = anthropic.Anthropic()


# ============================== YOUR PART =====================================


def agent_answer(
    question: str, retriever: Callable[..., list[dict]] = retrieve, top_k: int = 8
) -> dict:
    """Answer one question with the agent. Returns the answer plus what it did."""
    searches: list[str] = []  # every query Claude wrote
    sources: list[str] = []  # the source file of every chunk it got back

    # The tool is defined INSIDE agent_answer, so it can see (and fill) the two
    # lists above: a closure. Each question gets fresh lists.
    @beta_tool
    def search_docs(query: str) -> str:
        """Search the FastAPI docs. Call this for any FastAPI question, and rephrase
        the question into FastAPI terms if needed.

        Args:
            query: What to search for.
        """
        if len(searches) >= MAX_SEARCHES:
            raise ValueError(
                f"Search limit of {MAX_SEARCHES} reached. "
                "Answer with the documents you already have."
            )
        chunks = retriever(query, top_k=top_k)
        searches.append(query)
        sources.extend(c["source"] for c in chunks)
        return format_documents(chunks)

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=2048,
        max_iterations=MAX_STEPS,
        system=SYSTEM_PROMPT,
        tools=[search_docs],
        messages=[{"role": "user", "content": question}],
    )

    calls = input_tokens = output_tokens = 0
    for message in runner:  # one pass = one API call
        calls += 1
        for b in message.content:
            if b.type == "tool_use":
                print(f"  API call {calls}: search_docs({b.input['query']!r})")
        input_tokens += message.usage.input_tokens
        output_tokens += message.usage.output_tokens

    final = runner.until_done()
    if final.stop_reason != "end_turn":
        raise RuntimeError(f"Agent stopped early: {final.stop_reason}")

    return {
        "answer": "".join(b.text for b in final.content if b.type == "text"),
        "searches": searches,
        "sources": sources,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ================================ TRY IT ======================================

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    result = agent_answer(question)
    for query in result["searches"]:
        print(f"  searched: {query!r}")
    print(
        f"  {result['calls']} API calls, "
        f"{result['input_tokens']} in / {result['output_tokens']} out tokens\n"
    )
    print(result["answer"])

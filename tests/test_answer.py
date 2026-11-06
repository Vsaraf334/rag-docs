from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import answer as answer_module
from answer import MODEL, SYSTEM_PROMPT, answer, build_prompt

CHUNKS = [
    {"text": "Raise HTTPException.", "source": "errors.md", "distance": 0.5},
    {"text": "Use BackgroundTasks.", "source": "tasks.md", "distance": 0.9},
]


@pytest.fixture
def fake_client(monkeypatch):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hidden"),
            SimpleNamespace(type="text", text="Raise it. [errors.md]"),
        ]
    )
    monkeypatch.setattr(answer_module, "client", client)
    monkeypatch.setattr(answer_module, "retrieve", lambda question: CHUNKS)
    return client


def test_build_prompt_wraps_each_chunk_in_document_tags():
    prompt = build_prompt("How do I return a 404?", CHUNKS)
    assert '<document source="errors.md">\nRaise HTTPException.\n</document>' in prompt
    assert '<document source="tasks.md">\nUse BackgroundTasks.\n</document>' in prompt


def test_build_prompt_puts_documents_before_the_question():
    prompt = build_prompt("How do I return a 404?", CHUNKS)
    assert prompt.startswith("<documents>")
    assert prompt.endswith("Question: How do I return a 404?")


def test_answer_sends_system_prompt_and_retrieved_chunks(fake_client):
    answer("How do I return a 404?")
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["messages"] == [
        {"role": "user", "content": build_prompt("How do I return a 404?", CHUNKS)}
    ]


def test_answer_returns_only_text_blocks(fake_client):
    assert answer("How do I return a 404?") == "Raise it. [errors.md]"

"""Tests for the MCP server, through a real MCP client connected in-process.

Client(server) talks to the server object directly (no subprocess, no stdio), but
every call still goes through MCP: tool listing, argument validation, errors.
retrieve() is replaced with a fake, so no index or network is needed.
"""

import asyncio

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

import mcp_server

FAKE_CHUNKS = [
    {"text": "Use HTTPException to return errors.", "source": "handling-errors.md"},
    {"text": "Declare a BackgroundTasks parameter.", "source": "background-tasks.md"},
]


def text_of(result: CallToolResult) -> str:
    """A result's content can be text, an image, audio...; narrow it to text."""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def run(coro):
    """The MCP client is async; run one coroutine to completion."""
    return asyncio.run(coro)


async def list_tools():
    async with Client(mcp_server.server) as client:
        return (await client.list_tools()).tools


async def call(arguments: dict):
    async with Client(mcp_server.server) as client:
        return await client.call_tool("search_docs", arguments)


@pytest.fixture
def fake_retrieve(monkeypatch):
    """Replace retrieve() inside mcp_server, and record what it was called with."""
    calls = []

    def fake(query, top_k):
        calls.append((query, top_k))
        return FAKE_CHUNKS

    monkeypatch.setattr(mcp_server, "retrieve", fake)
    return calls


def test_lists_one_tool_with_descriptions():
    tools = run(list_tools())
    assert [t.name for t in tools] == ["search_docs"]
    tool = tools[0]
    assert tool.description and "FastAPI" in tool.description
    props = tool.input_schema["properties"]
    assert props["query"]["description"]  # the model needs to know what to send
    assert props["top_k"]["default"] == 8
    assert (props["top_k"]["minimum"], props["top_k"]["maximum"]) == (1, 20)
    assert tool.input_schema["required"] == ["query"]


def test_tool_is_marked_read_only():
    annotations = run(list_tools())[0].annotations
    assert annotations is not None
    assert annotations.read_only_hint is True  # it never changes anything
    assert annotations.open_world_hint is False  # a fixed local index, not the web


def test_search_returns_tagged_documents(fake_retrieve):
    result = run(call({"query": "404 error"}))
    assert not result.is_error
    text = text_of(result)
    assert '<document source="handling-errors.md">' in text
    assert "Use HTTPException to return errors." in text
    assert fake_retrieve == [("404 error", 8)]  # default top_k reached retrieve()


def test_top_k_is_passed_through(fake_retrieve):
    run(call({"query": "x", "top_k": 3}))
    assert fake_retrieve == [("x", 3)]


@pytest.mark.parametrize("top_k", [0, 21])
def test_top_k_out_of_range_is_rejected_before_searching(fake_retrieve, top_k):
    result = run(call({"query": "x", "top_k": top_k}))
    assert result.is_error
    assert "top_k" in text_of(result)
    assert fake_retrieve == []  # validation stopped it; retrieve never ran


def test_missing_query_is_rejected(fake_retrieve):
    result = run(call({}))
    assert result.is_error
    assert fake_retrieve == []


def test_empty_index_gives_a_helpful_error(monkeypatch):
    monkeypatch.setattr(mcp_server, "retrieve", lambda query, top_k: [])
    result = run(call({"query": "anything"}))
    assert result.is_error
    # ToolError: the model sees OUR message, not just "Error executing tool"
    assert "index is empty" in text_of(result)
    assert "ingest" in text_of(result)

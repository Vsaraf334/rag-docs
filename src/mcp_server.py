"""MCP server for rag-docs: lets any MCP host (Claude Code, Claude Desktop, ...) search the docs.

A host starts this file as a subprocess and talks to it over stdin/stdout, one
JSON message per line. You don't normally run it yourself.

Never print() in this file: stdout is the protocol channel, and a stray line
breaks the connection. (Logs go to stderr.)
"""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from answer import format_documents
from retrieve import retrieve

server = MCPServer(
    "rag-docs",
    instructions=(
        "Answer FastAPI questions using only the passages returned by search_docs. "
        "Cite the source file after each claim, like [body.md]. "
        "If the passages don't contain the answer, say you don't know."
    ),
)


@server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def search_docs(
    query: Annotated[
        str,
        Field(
            description="What to search for, in FastAPI terms, e.g. 'dependency injection Depends'."
        ),
    ],
    top_k: Annotated[
        int, Field(description="How many passages to return.", ge=1, le=20)
    ] = 8,
) -> str:
    """Search the FastAPI docs. Call this for any FastAPI question, using FastAPI terms."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        raise ToolError(
            "The document index is empty. Run `uv run python -m src.ingest` first."
        )
    return format_documents(chunks)


if __name__ == "__main__":
    server.run()

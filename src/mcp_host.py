"""A minimal MCP host: what Claude Code does with an MCP server, in ~50 lines.

1. Start the MCP server as a subprocess and ask for its tools (tools/list).
2. Convert each MCP tool into a Claude API tool definition.
3. Run the week 8 agent loop, turning every tool_use into an MCP tools/call.

The host knows nothing about Chroma, retrieve() or the docs. It only knows how to
start the server. Everything else it learns over MCP.

Run:  uv run --env-file .env python src/mcp_host.py "How do I return a 404 error?"
"""

import asyncio
import sys

import anthropic
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult, TextContent, Tool

MODEL = "claude-haiku-5-5"
MAX_STEPS = 6

# How to start the server: the same command you gave `claude mcp add`.
SERVER = StdioServerParameters(command=sys.executable, args=["src/mcp_server.py"])

claude = anthropic.AsyncAnthropic()  # async version: every call is awaited


# ============================== YOUR PART =====================================


def to_claude_tool(tool: Tool) -> ToolParam:
    """MCP tool definition -> Claude API tool definition (doc 1.4: almost 1-to-1)."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,  # MCP calls it inputSchema; the SDK object: input_schema
    }


def to_tool_result(tool_use_id: str, result: CallToolResult) -> ToolResultBlockParam:
    """MCP tools/call result -> Claude API tool_result block."""
    text = "".join(c.text for c in result.content if isinstance(c, TextContent))
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": result.is_error,  # MCP's isError, as the SDK object field
    }


async def run_host(question: str) -> str:
    async with Client(SERVER) as mcp:  # starts the server; stops it on exit
        listed = await mcp.list_tools()  # step 1: tools/list
        tools = [to_claude_tool(t) for t in listed.tools]
        print(f"  tools from the server: {[t['name'] for t in tools]}")

        messages: list[MessageParam] = [{"role": "user", "content": question}]
        for step in range(1, MAX_STEPS + 1):
            response = await claude.messages.create(
                model=MODEL,
                max_tokens=2048,
                tools=tools,
                messages=messages,
                system=mcp.instructions or "",
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason != "tool_use":
                raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

            results = []
            for b in response.content:
                if b.type == "tool_use":
                    print(f"  step {step}: {b.name}({b.input})")
                    result = await mcp.call_tool(b.name, b.input)  # step 3: tools/call
                    results.append(to_tool_result(b.id, result))
            messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")


# ================================ TRY IT ======================================

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I make a query parameter required?"
    print(asyncio.run(run_host(question)))

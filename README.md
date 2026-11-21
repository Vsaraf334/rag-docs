# rag-docs

A retrieval-augmented generation (RAG) system that answers questions about the FastAPI tutorial and cites its sources. When the documents don't contain an answer, it says so instead of guessing.

Every change to the system is measured with an evaluation suite: 24 questions, retrieval scored separately from answers, and answers graded by an LLM judge. The fixed pipeline answers **19 of 20** answerable questions correctly and declines **4 of 4** questions the documents can't answer.

It can run in two modes:
- **Pipeline:** search once with the user's words, then answer. One API call.
- **Agent:** search is a tool, so Claude writes its own queries and can search again. It answers **20 of 20** in every run, at about **2.3× the cost** ([results](#agent-vs-pipeline)).

Chunks can be stored in Chroma or in Postgres with pgvector. The two backends return identical results for all 24 eval questions.

The search is also available as an **MCP server**, so other apps such as Claude Code can search the docs ([details](#use-it-from-claude-code-mcp)).

Built with Python, Postgres + pgvector (via Docker Compose), Chroma, and the Anthropic Python SDK (Claude Haiku 5.5 for answers, Claude Sonnet 5.5 as the judge).

## Example

```
$ uv run --env-file .env python src/answer.py "How do I run code after sending the response?"
You can use background tasks, which run after the response is returned. [background-tasks.md]
1. Declare a `BackgroundTasks` parameter in your path operation function. [background-tasks.md]
2. Create a task function ...
3. Call `.add_task()` with the function and its arguments. [background-tasks.md]

$ uv run --env-file .env python src/answer.py "How do I connect FastAPI to MongoDB?"
I don't know. The provided documents cover ... They don't mention MongoDB or how to
connect FastAPI to a database, so I can't answer this from them.
```

## How it works

```
Indexing (once)
  FastAPI docs ─► prepare_docs.py ─► data/*.md ─► chunk_text() ─► embeddings ─► Chroma (chroma_db/)
                  inline the code examples        800 characters   all-MiniLM-L6-v2   or Postgres + pgvector

Answering, pipeline mode (per question)
  question ─► retrieve() ─► 8 closest chunks ─► Claude Haiku ─► answer with [source] citations

Answering, agent mode (per question)
  question ─► Claude Haiku ─► search_docs("Enum path parameter") ─► retrieve() ─► 8 chunks ─┐
                   ▲                                                                       │
                   └──────────── reads the results, searches again (max 2) ◄───────────────┘
              ─► answer with [source] citations
```

| File | What it does |
| --- | --- |
| `src/prepare_docs.py` | Copies 8 FastAPI tutorial pages into `data/`, replacing `{* ../../docs_src/... *}` placeholders with the real code |
| `src/ingest.py` | Loads documents, splits them into overlapping chunks, and stores them in Chroma with their source file |
| `src/retrieve.py` | Finds the chunks closest in meaning to a question (Chroma) |
| `src/pg_store.py` | The Postgres backend: stores chunks with their embeddings and searches with `ORDER BY embedding <=> question LIMIT k` |
| `compose.yaml` | Runs Postgres with pgvector locally, with a health check |
| `src/answer.py` | Sends the chunks to Claude in `<document source="...">` tags and asks for a cited answer |
| `src/agent.py` | The agent: `retrieve()` exposed as a `search_docs` tool through the SDK's tool runner, with limits on searches and steps |
| `src/mcp_server.py` | An MCP server exposing `search_docs` (read-only, `top_k` validated, with usage instructions for hosts) |
| `src/mcp_host.py` | A minimal MCP host: starts the server, converts its tools into Claude API tools, and runs the agent loop over MCP |
| `evals/questions.jsonl` | 24 questions, each with its correct source file and the key fact a good answer must contain |
| `evals/judge.py` | An LLM judge that grades an answer as `correct`, `partial`, or `incorrect`, with a one-sentence reason |
| `evals/run_eval.py` | Runs every question in either mode and reports retrieval hit rate, answer accuracy, correct refusals, and the cost: API calls, searches, tokens and seconds per question |

## Evaluation

The eval measures the two halves of RAG separately, because they fail for different reasons and need different fixes:

- **Retrieval hit rate:** is the correct source file among the retrieved chunks?
- **Answer accuracy:** does the judge rate the answer `correct`?
- **Correct refusals:** for the 4 questions the documents can't answer (MongoDB, JWT, Kubernetes, CORS), does the system say it doesn't know?

### Results

| # | Change | Retrieval | Answers | Refusals |
| --- | --- | --- | --- | --- |
| 1 | Baseline: 800-character chunks, top 4 chunks, substring grading | 18/20 | 18/20 | 4/4 |
| 2 | Grade answers with an LLM judge instead of a substring check | 18/20 | 16/20 | 4/4 |
| 3 | Inline the code examples into the documents | 17/20 | 17/20 | 4/4 |
| 4 | Retrieve 8 chunks instead of 4 | **19/20** | **19/20** | **4/4** |
| 5 | Move storage from Chroma to Postgres + pgvector | 19/20 | 18/20 | 4/4 |
| 6 | Hybrid search: vector + keyword search, merged with RRF (not adopted) | 19/20 | 18/20 | 4/4 |
| 7 | Agent mode: Claude writes the search queries, up to 5 searches | **20/20** | **20/20** | **4/4** |
| 8 | Agent mode with at most 2 searches (same quality, less cost) | **20/20** | **20/20** | **4/4** |

What each step showed:

1. **The substring grader was wrong in both directions.** It failed a correct answer that described the output-model pattern without naming `UserOut`. It also passed answers that contained the expected word but said "I don't know how". Switching to an LLM judge lowered the score but made it honest.
2. **The judge's reasons pointed to one root cause.** Three answers said "the documents don't include the code". The FastAPI docs reference their examples through placeholders, so the code never reached the index. Inlining it fixed all three.
3. **The remaining failures were all retrieval misses.** In those cases the right file never reached Claude, and Claude correctly declined instead of guessing. Raising `top_k` from 4 to 8 fixed two of the three. It doubles the input tokens per answer, and it introduced no new hallucinations.
4. **The Postgres migration changed nothing in retrieval.** Both backends returned the same chunks in the same order for all 24 questions. The one-question difference in answers comes from LLM variation between runs, not from the migration: the chunks were identical.
5. **Hybrid search made things worse.** It kept the hit rate at 19/20 but swapped one miss for another, and keyword matches on common words such as "response" pushed useful chunks out. Vector search stays the default. The full write-up is in [docs/hybrid-search.md](docs/hybrid-search.md), and the code is on the `experiment/hybrid-search` branch.
6. **The agent fixed the vocabulary-mismatch miss.** The user's words "fixed set of allowed values" never found the `Enum` section. The agent searched for `'Enum path parameter predefined values'` (the docs' own words, from Claude's knowledge of FastAPI) and found it in every run. The pipeline's other missing answer varied between runs (noise), so the real gain is one question.
7. **Unanswerable questions caused most of the agent's extra cost.** It refused all 4 correctly, but only after 4–5 rephrased searches each: up to 30,000 input tokens for one question, against the pipeline's 2,500. Limiting it to 2 searches kept every score and cut the worst case from 30k to 12k tokens.

### Agent vs pipeline

Chroma, `top_k = 8`, Claude Haiku 5.5. Pipeline: two runs. Agent: four runs (search limits of 5, 5, 3 and 2), and every one scored 20/20 and 4/4. The cost column is the run with a limit of 2:

| | Pipeline | Agent (max 2 searches) |
| --- | --- | --- |
| Retrieval hit rate | 19/20 | **20/20** |
| Answer accuracy | 18–19/20 | **20/20** |
| Correct refusals | 4/4 | 4/4 |
| API calls per question | 1 | 2.2 (max 3) |
| Input tokens per question, average / worst | 2,485 / 2,658 | 5,691 / 12,254 |
| Seconds per question | 2.8 | 4.2 |
| Answer cost for all 24 questions | $0.011 | $0.019 |

Answerable questions cost about 1.8× the pipeline's tokens. In the first run's per-question details, every one took exactly two API calls: search (once, or twice in parallel), then answer. The unanswerable questions cost about 5× more, because the agent keeps rephrasing until it hits the search limit.

**Verdict:** the agent is more accurate by one question and costs about 2.3× more on average, and 4.6× more in the worst case. For a low-volume docs assistant that's a good trade. For a high-volume service, a single query-rewriting step before a fixed search might get most of the gain for much less.

Run it yourself:

```bash
uv run --env-file .env python evals/run_eval.py                  # pipeline, Chroma, top_k = 8
uv run --env-file .env python evals/run_eval.py --mode agent     # the agent
uv run --env-file .env python evals/run_eval.py --backend pg     # Postgres (works with either mode)
uv run --env-file .env python evals/run_eval.py --top-k 4        # compare with 4
```

Per-question details (answers, searches, tokens, verdicts) are written to `evals/results-<mode>-<backend>.jsonl`.

## Running locally

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

```bash
git clone https://github.com/Vsaraf334/rag-docs.git
cd rag-docs
uv sync
echo "ANTHROPIC_API_KEY=your-key-here" > .env

uv run python -m src.ingest                                    # build the index
uv run --env-file .env python src/answer.py "How do I return a 404 error?"
uv run --env-file .env python src/agent.py "How do I return a 404 error?"    # agent mode
```

The prepared documents are already in `data/`. To regenerate them, clone the FastAPI repository with `docs/en/docs/tutorial` and `docs_src` to `~/Documents/fastapi-src`, then run `uv run python src/prepare_docs.py`. After changing `data/`, delete `chroma_db/` and re-run ingest, so chunks from older versions of a document don't remain in the index.

`.env` is git-ignored. Never commit an API key.

### With Postgres

Requires Docker.

```bash
docker compose up -d --wait                                      # Postgres + pgvector on localhost:5432
echo "DATABASE_URL=postgresql://postgres:localdev@localhost:5432/postgres" >> .env
uv run --env-file .env python src/pg_store.py                    # embed and store the chunks
uv run --env-file .env python evals/run_eval.py --backend pg
```

The database listens on `127.0.0.1` only, and its data lives in a Docker volume, so it survives `docker compose down`. Use `docker compose down -v` to delete it.

### Use it from Claude Code (MCP)

`src/mcp_server.py` exposes the search as an [MCP](https://modelcontextprotocol.io) server. Any MCP host can start it and call `search_docs`. To add it to Claude Code for all your projects:

```bash
claude mcp add --scope user rag-docs -- uv run --directory /path/to/rag-docs python src/mcp_server.py
claude mcp list                                                  # rag-docs: ... ✔ Connected
```

Then ask Claude Code a FastAPI question in a new session: it searches these docs and cites files like `[query-params.md]`. The server needs no API key, because the host provides the model.

`src/mcp_host.py` shows what a host does, in about 50 lines: it starts the server, lists its tools, converts them to Claude API tool definitions, and runs the agent loop, turning each `tool_use` into an MCP `tools/call`:

```bash
uv run --env-file .env python src/mcp_host.py "How do I return a 404 error?"
```

## Tests

```bash
uv run pytest     # 36 tests, no API key needed
```

The tests cover chunk boundaries, document loading, storing and re-storing chunks without duplicates, retrieval order, prompt construction, and the Claude call itself (with a mocked client). The Chroma tests use a temporary directory, so they never touch the real index.

8 tests cover the MCP server through a real MCP client connected in-process: the tool listing and its descriptions, searching, `top_k` validation, a missing query, the read-only annotation, and the error message for an empty index.

7 of the tests are integration tests that run against the real Postgres. They use a separate `rag_test` database and empty it before each test, and one of them checks that Postgres and Chroma rank chunks the same way. If Postgres isn't running, they're skipped with a message instead of failing.

## Design decisions

- **Answer only from the documents, cite every claim, and say "I don't know" otherwise.** A RAG system is only useful if you can trust where an answer came from. The refusal behaviour is measured on every eval run.
- **Documents go in XML tags.** `<document source="...">` separates the data from the instructions and gives Claude the file name to cite.
- **Chunk ids are deterministic (`body.md:3`) and storage uses upsert.** Re-running ingest updates the index instead of duplicating it.
- **The eval retrieves once and passes the same chunks to Claude.** The retrieval score and the answer score always describe the same run.
- **A stronger model judges a smaller one.** Haiku answers and Sonnet grades. The judge returns JSON that is validated with Pydantic, and its reasons are spot-checked by hand.
- **Postgres instead of a separate vector database.** Most teams already run Postgres. With pgvector, chunks can sit next to normal application data and be filtered with ordinary SQL. The migration was checked with the eval: identical retrieval on every question.
- **The agent is bounded in code, not by the model's judgment.** It has at most 2 searches and 6 API calls per question, and `max_tokens` on every call. A refused search goes back to Claude as an error result ("Search limit reached. Answer with the documents you already have"), so it still answers. The tool runner has no step limit by default and doesn't raise when it stops early, so the code checks `stop_reason` itself.
- **Claude's knowledge for searching, the documents for answering.** The agent may use what it knows about FastAPI to choose search words. The answer must still come only from the retrieved documents, and the judge checks it.
- **Log what the agent did; don't ask it.** The `search_docs` tool records every query and source through a closure. In one run, Claude's answer said it had searched three times when it had searched four.
- **The MCP server is a thin adapter.** It reuses `retrieve()` and `format_documents()` unchanged, so the pipeline, the agent and MCP hosts all search the same way.
- **The server offers; the host decides.** Asked about MongoDB, the MCP host with no system prompt answered from Claude's general knowledge. With the server's `instructions` used as its system prompt, it refused. Grounding rules are behaviour, and behaviour belongs to the host. A host should only adopt instructions from servers it trusts, since they become part of its system prompt.
- **Measure cost, not just quality.** The eval reports API calls, tokens and seconds per question, as averages and worst cases, because agents trade cost for accuracy.
- **Measure before adding complexity.** Hybrid search is common advice. I built it (vector and keyword rankings fused with Reciprocal Rank Fusion in a single SQL query), measured it, found it hurt two questions, and kept plain vector search. The experiment is documented in [docs/hybrid-search.md](docs/hybrid-search.md).

## Limitations

- **24 questions is a small eval.** One question is 5%, and LLM answers vary between runs, so a one-question difference is within noise.
- **Retrieval is scored per file, not per section.** A chunk from the right file but the wrong section still counts as a hit. The hybrid search experiment showed this hiding real problems.
- **One known pipeline miss:** "How do I restrict a URL parameter to a fixed set of allowed values?" never retrieves the `Enum` section of `path-params.md`. The question's words ("fixed set of allowed values") don't match the docs' words ("predefined values", "Enum"). Agent mode fixes it.
- **The agent's behaviour varies between runs.** The same question can take 2 searches in one run and 4 in the next. The cost figures come from single runs.
- **No unit tests for the agent yet.** It's covered by the eval, not by `pytest`.
- **Character-based chunking** can split a code block or sentence in half.
- **The embedding model** (all-MiniLM-L6-v2) is small and reads at most about 256 tokens per chunk.

## Next steps

- A single query-rewriting step before a fixed search: does it get the agent's accuracy at close to the pipeline's cost?
- Unit tests for the agent, with a mocked tool runner
- Score retrieval per section instead of per file
- Heading-aware chunking, so each chunk is one section
- A larger eval set, run several times and averaged
- An HNSW index and a connection pool, for larger data and a web service

## Credits

Sample documents: FastAPI tutorial pages from [fastapi/fastapi](https://github.com/fastapi/fastapi) (MIT licence).

# rag-docs

A retrieval-augmented generation (RAG) system that answers questions about the FastAPI tutorial and cites its sources. When the documents don't contain an answer, it says so instead of guessing.

Every change to the system is measured with an evaluation suite: 24 questions, retrieval scored separately from answers, and answers graded by an LLM judge. The current version answers **19 of 20** answerable questions correctly and declines **4 of 4** questions the documents can't answer.

Built with Python, Chroma, and the Anthropic Python SDK (Claude Haiku 5.5 for answers, Claude Sonnet 5.5 as the judge).

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
                  inline the code examples        800 characters   all-MiniLM-L6-v2

Answering (per question)
  question ─► retrieve() ─► 8 closest chunks ─► Claude Haiku ─► answer with [source] citations
```

| File | What it does |
| --- | --- |
| `src/prepare_docs.py` | Copies 8 FastAPI tutorial pages into `data/`, replacing `{* ../../docs_src/... *}` placeholders with the real code |
| `src/ingest.py` | Loads documents, splits them into overlapping chunks, and stores them in Chroma with their source file |
| `src/retrieve.py` | Finds the chunks closest in meaning to a question |
| `src/answer.py` | Sends the chunks to Claude in `<document source="...">` tags and asks for a cited answer |
| `evals/questions.jsonl` | 24 questions, each with its correct source file and the key fact a good answer must contain |
| `evals/judge.py` | An LLM judge that grades an answer as `correct`, `partial`, or `incorrect`, with a one-sentence reason |
| `evals/run_eval.py` | Runs every question and reports retrieval hit rate, answer accuracy, and correct refusals |

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

What each step showed:

1. **The substring grader was wrong in both directions.** It failed a correct answer that described the output-model pattern without naming `UserOut`. It also passed answers that contained the expected word but said "I don't know how". Switching to an LLM judge lowered the score but made it honest.
2. **The judge's reasons pointed to one root cause.** Three answers said "the documents don't include the code". The FastAPI docs reference their examples through placeholders, so the code never reached the index. Inlining it fixed all three.
3. **The remaining failures were all retrieval misses.** In those cases the right file never reached Claude, and Claude correctly declined instead of guessing. Raising `top_k` from 4 to 8 fixed two of the three. It doubles the input tokens per answer, and it introduced no new hallucinations.

Run it yourself:

```bash
uv run --env-file .env python evals/run_eval.py        # top_k = 8
uv run --env-file .env python evals/run_eval.py 4      # compare with 4
```

## Running locally

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

```bash
git clone https://github.com/Vsaraf334/rag-docs.git
cd rag-docs
uv sync
echo "ANTHROPIC_API_KEY=your-key-here" > .env

uv run python -m src.ingest                                    # build the index
uv run --env-file .env python src/answer.py "How do I return a 404 error?"
```

The prepared documents are already in `data/`. To regenerate them, clone the FastAPI repository with `docs/en/docs/tutorial` and `docs_src` to `~/Documents/fastapi-src`, then run `uv run python src/prepare_docs.py`. After changing `data/`, delete `chroma_db/` and re-run ingest, so chunks from older versions of a document don't remain in the index.

`.env` is git-ignored. Never commit an API key.

## Tests

```bash
uv run pytest     # 21 tests, no API key or network needed
```

The tests cover chunk boundaries, document loading, storing and re-storing chunks without duplicates, retrieval order, prompt construction, and the Claude call itself (with a mocked client). The Chroma tests use a temporary directory, so they never touch the real index.

## Design decisions

- **Answer only from the documents, cite every claim, and say "I don't know" otherwise.** A RAG system is only useful if you can trust where an answer came from. The refusal behaviour is measured on every eval run.
- **Documents go in XML tags.** `<document source="...">` separates the data from the instructions and gives Claude the file name to cite.
- **Chunk ids are deterministic (`body.md:3`) and storage uses upsert.** Re-running ingest updates the index instead of duplicating it.
- **The eval retrieves once and passes the same chunks to Claude.** The retrieval score and the answer score always describe the same run.
- **A stronger model judges a smaller one.** Haiku answers and Sonnet grades. The judge returns JSON that is validated with Pydantic, and its reasons are spot-checked by hand.
- **Measure before adding complexity.** I considered hybrid search (BM25 + embeddings). A check showed vector search already ranks exact identifiers well on this data, and the remaining miss is a vocabulary mismatch that keyword search can't fix, so I left it out.

## Limitations

- **24 questions is a small eval.** One question is 5%, so small differences between runs can be noise.
- **One known miss:** "How do I restrict a URL parameter to a fixed set of allowed values?" never retrieves the `Enum` section of `path-params.md`. The question's words ("fixed set of allowed values") don't match the docs' words ("predefined values", "Enum").
- **Character-based chunking** can split a code block or sentence in half.
- **The embedding model** (all-MiniLM-L6-v2) is small and reads at most about 256 tokens per chunk.

## Next steps

- Query rewriting or a reranker for the vocabulary-mismatch miss
- Postgres with pgvector, and hybrid search through Postgres full-text search
- Heading-aware chunking, so each chunk is one section
- A larger eval set

## Credits

Sample documents: FastAPI tutorial pages from [fastapi/fastapi](https://github.com/fastapi/fastapi) (MIT licence).

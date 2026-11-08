"""Store and search chunks in Postgres with pgvector."""

import os

import psycopg
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pgvector.psycopg import register_vector

from ingest import build_chunks, load_documents

embed = DefaultEmbeddingFunction()  # all-MiniLM-L6-v2, the same model Chroma used

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,
    source    text NOT NULL,
    text      text NOT NULL,
    embedding vector(384) NOT NULL
);
-- Keyword search: Postgres keeps the stemmed words of each chunk up to date itself
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS words tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_words_idx ON chunks USING gin (words);
"""

UPSERT = """
INSERT INTO chunks (id, source, text, embedding) VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE
SET source = EXCLUDED.source, text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""

SEARCH = """
SELECT text, source, embedding <=> %s AS distance
FROM chunks
ORDER BY distance
LIMIT %s
"""

# Reciprocal Rank Fusion of vector search and keyword search, in one query.
# The keyword query ORs the question's words: plainto_tsquery would require all of them.
HYBRID_SEARCH = """
WITH vector AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s) AS rank
    FROM chunks
    ORDER BY embedding <=> %(vector)s
    LIMIT %(candidates)s
),
keyword AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(words, query) DESC) AS rank
    FROM chunks,
         (SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')::tsquery
          AS query) AS q
    WHERE words @@ query
    ORDER BY ts_rank_cd(words, query) DESC
    LIMIT %(candidates)s
)
SELECT c.text, c.source,
       coalesce(1.0 / (%(rrf_k)s + v.rank), 0)
     + coalesce(1.0 / (%(rrf_k)s + k.rank), 0) AS score
FROM vector v
FULL OUTER JOIN keyword k ON v.id = k.id
JOIN chunks c ON c.id = coalesce(v.id, k.id)
ORDER BY score DESC
LIMIT %(top_k)s
"""
RRF_K = 60  # standard constant: stops the 1st place from dominating the score
CANDIDATES = 50  # how many results each search contributes before fusing


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute(SCHEMA)
    register_vector(conn)  # teach psycopg the vector type (needs the extension first)
    return conn


def store_chunks_pg(conn: psycopg.Connection, chunks: list[dict]) -> None:
    vectors = embed([c["text"] for c in chunks])
    rows = [(c["id"], c["source"], c["text"], v) for c, v in zip(chunks, vectors)]
    with conn.cursor() as cur:
        cur.executemany(UPSERT, rows)


def retrieve_pg(question: str, top_k: int = 8) -> list[dict]:
    """Same contract as retrieve(): the top_k closest chunks, closest first."""
    query_vector = embed([question])[0]
    with connect() as conn:
        rows = conn.execute(SEARCH, (query_vector, top_k)).fetchall()
    return [{"text": t, "source": s, "distance": d} for t, s, d in rows]


def retrieve_hybrid_pg(question: str, top_k: int = 8) -> list[dict]:
    """Like retrieve_pg(), but fuses vector and keyword search. Higher score = better."""
    params = {
        "vector": embed([question])[0],
        "question": question,
        "candidates": CANDIDATES,
        "rrf_k": RRF_K,
        "top_k": top_k,
    }
    with connect() as conn:
        rows = conn.execute(HYBRID_SEARCH, params).fetchall()
    return [{"text": t, "source": s, "score": float(sc)} for t, s, sc in rows]


if __name__ == "__main__":
    with connect() as conn:
        store_chunks_pg(conn, build_chunks(load_documents()))
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        assert row
        print(f"{row[0]} chunks in Postgres")

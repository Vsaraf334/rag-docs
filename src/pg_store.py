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


if __name__ == "__main__":
    with connect() as conn:
        store_chunks_pg(conn, build_chunks(load_documents()))
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        assert row
        print(f"{row[0]} chunks in Postgres")

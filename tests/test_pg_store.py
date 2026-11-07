"""Integration tests for the Postgres backend.

They need a running Postgres (`docker compose up -d --wait`) and are skipped otherwise.
They use a separate `rag_test` database, so the real `chunks` table is never touched.
"""

import os

import psycopg
import pytest

from ingest import build_chunks, store_chunks
from pg_store import connect, retrieve_pg, store_chunks_pg
from retrieve import retrieve

ADMIN_URL = os.environ.get(
    "TEST_ADMIN_URL", "postgresql://postgres:localdev@localhost:5432/postgres"
)
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/rag_test"

DOCS = [
    ("errors.md", "Raise HTTPException for a 404 error."),
    ("food.md", "Bake banana bread at 180 degrees."),
    ("travel.md", "The Eiffel Tower is in Paris."),
]


@pytest.fixture(scope="module")
def test_database():
    """Create the rag_test database once per test file, or skip if Postgres is down."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip(
            "Postgres is not running; start it with: docker compose up -d --wait"
        )
    with admin:
        if not admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'rag_test'"
        ).fetchone():
            admin.execute("CREATE DATABASE rag_test")
    return TEST_URL


@pytest.fixture
def conn(test_database, monkeypatch):
    """A connection to rag_test with an empty chunks table."""
    monkeypatch.setenv("DATABASE_URL", test_database)
    with connect() as conn:
        conn.execute("TRUNCATE chunks")
        yield conn


def count(conn) -> int:
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert row
    return row[0]


def test_store_twice_does_not_duplicate(conn):
    chunks = build_chunks(DOCS)
    store_chunks_pg(conn, chunks)
    store_chunks_pg(conn, chunks)
    assert count(conn) == 3


def test_store_again_updates_changed_text(conn):
    store_chunks_pg(conn, build_chunks([("a.md", "old text")]))
    store_chunks_pg(conn, build_chunks([("a.md", "new text")]))
    row = conn.execute("SELECT text FROM chunks WHERE id = 'a.md:0'").fetchone()
    assert row == ("new text",)


def test_retrieve_returns_closest_chunk_first_with_source(conn):
    store_chunks_pg(conn, build_chunks(DOCS))
    results = retrieve_pg("item not found", top_k=3)
    assert results[0]["source"] == "errors.md"
    assert len(results) == 3
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    assert all(isinstance(d, float) for d in distances)


def test_retrieve_respects_top_k(conn):
    store_chunks_pg(conn, build_chunks(DOCS))
    assert len(retrieve_pg("item not found", top_k=1)) == 1


@pytest.mark.parametrize(
    "question", ["item not found", "dessert", "landmarks in France"]
)
def test_ranks_chunks_like_chroma(conn, tmp_path, question):
    chunks = build_chunks(DOCS)
    store_chunks_pg(conn, chunks)
    store_chunks(chunks, path=tmp_path)
    chroma_order = [r["source"] for r in retrieve(question, top_k=3, path=tmp_path)]
    pg_order = [r["source"] for r in retrieve_pg(question, top_k=3)]
    assert pg_order == chroma_order

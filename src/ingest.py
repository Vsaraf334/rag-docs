"""Load documents from data/, split them into chunks, embed them, and store them.

Week 4 task: implement load_documents, chunk_text, and store_chunks.
"""

from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "docs"
CHUNK_SIZE = 800  # characters per chunk; tune this during evals
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks
EXCLUDED_FILES = {"README.md"}  # files in data/ that are not documents to index


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """Return (filename, text) pairs for every .md and .txt file in data_dir."""
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix in {".md", ".txt"} and path.name not in EXCLUDED_FILES:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping character windows."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if not text:
        return []

    chunks = []
    start = 0
    while True:
        chunks.append(text[start : start + size])
        if start + size >= len(text):  # this chunk reached the end
            return chunks
        start += size - overlap


def build_chunks(docs: list[tuple[str, str]]) -> list[dict]:
    """Turn (filename, text) pairs into chunk dicts with id, text and source."""
    chunks = []
    for filename, text in docs:
        for i, piece in enumerate(chunk_text(text)):
            chunks.append({"id": f"{filename}:{i}", "text": piece, "source": filename})
    return chunks


def get_collection(path: Path = CHROMA_DIR):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks: list[dict], path: Path = CHROMA_DIR) -> None:
    """Embed and store chunks. Re-running replaces chunks with the same id."""
    col = get_collection(path)
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )


if __name__ == "__main__":
    chunks = build_chunks(load_documents())
    store_chunks(chunks)
    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR}")

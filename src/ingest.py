"""Load documents from data/, split them into chunks, embed them, and store them.

Week 4 task: implement load_documents, chunk_text, and store_chunks.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_SIZE = 800  # characters per chunk; tune this during evals
CHUNK_OVERLAP = 100  # characters shared between neighbouring chunks


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """Return (filename, text) pairs for every .md and .txt file in data_dir."""
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix in {".md", ".txt"}:
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


def store_chunks(chunks: list[dict]) -> None:
    """Embed and store chunks. Implement with Chroma or pgvector in week 4."""
    raise NotImplementedError


if __name__ == "__main__":
    for name, text in load_documents():
        print(f"{name}: {len(chunk_text(text))} chunks")

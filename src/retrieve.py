"""Retrieve the most relevant chunks for a question."""

import sys
from pathlib import Path

from ingest import CHROMA_DIR, get_collection


def retrieve(question: str, top_k: int = 8, path: Path = CHROMA_DIR) -> list[dict]:
    """Return the top_k closest chunks as dicts with text, source and distance."""
    res = get_collection(path).query(query_texts=[question], n_results=top_k)
    docs, metas, dists = res["documents"], res["metadatas"], res["distances"]
    assert docs and metas and dists  # always included by query()'s defaults
    return [
        {"text": text, "source": meta["source"], "distance": dist}
        for text, meta, dist in zip(docs[0], metas[0], dists[0])
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    for chunk in retrieve(question):
        print(f"{chunk['distance']:.3f}  {chunk['source']}")
        print("   ", chunk["text"][:100].replace("\n", " "))

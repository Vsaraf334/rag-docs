from ingest import build_chunks, store_chunks
from retrieve import retrieve


def make_index(path):
    docs = [
        ("errors.md", "Raise HTTPException for a 404 error."),
        ("food.md", "Bake banana bread at 180 degrees."),
        ("travel.md", "The Eiffel Tower is in Paris."),
    ]
    store_chunks(build_chunks(docs), path=path)


def test_returns_closest_chunk_first_with_source(tmp_path):
    make_index(tmp_path)
    results = retrieve("item not found", top_k=1, path=tmp_path)
    assert results == [
        {
            "text": "Raise HTTPException for a 404 error.",
            "source": "errors.md",
            "distance": results[0]["distance"],
        }
    ]
    assert isinstance(results[0]["distance"], float)


def test_returns_top_k_sorted_by_distance(tmp_path):
    make_index(tmp_path)
    results = retrieve("item not found", top_k=3, path=tmp_path)
    distances = [r["distance"] for r in results]
    assert len(results) == 3
    assert distances == sorted(distances)

from ingest import build_chunks, get_collection, store_chunks


def test_build_chunks_ids_and_source():
    docs = [("a.md", "short text"), ("b.md", "x" * 1500)]
    chunks = build_chunks(docs)
    assert [c["id"] for c in chunks] == ["a.md:0", "b.md:0", "b.md:1"]
    assert all(c["source"] == c["id"].split(":")[0] for c in chunks)


def test_build_chunks_skips_empty_documents():
    assert build_chunks([("empty.md", "")]) == []


def test_store_chunks_twice_does_not_duplicate(tmp_path):
    chunks = build_chunks([("a.md", "Raise HTTPException for a 404 error.")])
    store_chunks(chunks, path=tmp_path)
    store_chunks(chunks, path=tmp_path)
    assert get_collection(tmp_path).count() == 1


def test_stored_chunk_is_searchable_with_source(tmp_path):
    docs = [
        ("errors.md", "Raise HTTPException for a 404 error."),
        ("food.md", "Bake banana bread at 180 degrees."),
    ]
    store_chunks(build_chunks(docs), path=tmp_path)
    res = get_collection(tmp_path).query(query_texts=["item not found"], n_results=1)
    assert res["metadatas"][0][0] == {"source": "errors.md"}

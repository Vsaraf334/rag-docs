import pytest

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text


def test_text_shorter_than_chunk_size_is_one_chunk():
    text = "a" * (CHUNK_SIZE - 1)
    assert chunk_text(text) == [text]


def test_text_shorter_than_overlap_size_is_one_chunk():
    text = "a" * (CHUNK_OVERLAP - 1)
    assert chunk_text(text) == [text]


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []


def test_text_exactly_overlap_size_is_one_chunk():
    text = "a" * CHUNK_OVERLAP
    assert chunk_text(text) == [text]


def test_no_duplicate_tail_chunk():
    # 1450 chars: chunk 2 (700..1450) reaches the end, so there must not be a third.
    assert len(chunk_text("a" * 1450)) == 2


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_neighbouring_chunks_share_overlap_characters():
    text = "".join(chr(97 + i % 26) for i in range(2000))
    chunks = chunk_text(text)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev[-CHUNK_OVERLAP:] == nxt[:CHUNK_OVERLAP]


def test_chunks_cover_whole_text_and_respect_size():
    text = "".join(chr(97 + i % 26) for i in range(2500))
    chunks = chunk_text(text)

    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # Rebuild the text by dropping the overlap from every chunk after the first.
    rebuilt = chunks[0] + "".join(c[CHUNK_OVERLAP:] for c in chunks[1:])
    assert rebuilt == text

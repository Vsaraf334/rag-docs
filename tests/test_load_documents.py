from ingest import load_documents


def test_loads_md_and_txt_files_in_name_order(tmp_path):
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")

    assert load_documents(tmp_path) == [("a.md", "alpha"), ("b.txt", "bravo")]


def test_skips_readme(tmp_path):
    (tmp_path / "README.md").write_text("Put your documents here.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("real content", encoding="utf-8")

    assert load_documents(tmp_path) == [("notes.md", "real content")]


def test_skips_other_file_types(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")

    assert load_documents(tmp_path) == []

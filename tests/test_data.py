import json

import pytest

from finsearch.data import load_live_items, load_questions


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_folder_maps_files_to_search_classes(tmp_path):
    _write(tmp_path / "historical-lookup.jsonl", [{"id": "h1", "question": "Q?", "answer": "A", "tolerance": "exact"}])
    _write(tmp_path / "multi-hop-lookup.jsonl", [{"id": "m1", "question": "Q2?", "answer": "B", "tolerance": ""}])
    qs = {q["id"]: q for q in load_questions(str(tmp_path))}
    assert qs["h1"]["search"] == "single" and qs["h1"]["task"] == "historical-lookup"
    assert qs["m1"]["search"] == "multi" and qs["m1"]["task"] == "multi-hop-lookup"
    assert [q["id"] for q in load_questions(str(tmp_path), ("multi-hop-lookup",))] == ["m1"]


def test_single_file_needs_a_class(tmp_path):
    f = tmp_path / "x.jsonl"
    _write(f, [{"id": "a", "question": "Q", "answer": "A", "search": "multi"}])
    assert load_questions(str(f))[0]["task"] == "multi-hop-lookup"
    _write(f, [{"id": "a", "question": "Q", "answer": "A"}])
    with pytest.raises(ValueError):
        load_questions(str(f))


def test_duplicate_ids_rejected(tmp_path):
    f = tmp_path / "x.jsonl"
    _write(f, [{"id": "a", "question": "Q", "answer": "A", "search": "single"}] * 2)
    with pytest.raises(ValueError):
        load_questions(str(f))


def test_live_items_reject_unknown_assets(tmp_path):
    f = tmp_path / "live-lookup.jsonl"
    _write(f, [{"id": "x", "asset": "fx", "name": "EUR", "ticker": "EUR", "query_template": "q"}])
    with pytest.raises(ValueError):
        load_live_items(str(tmp_path))

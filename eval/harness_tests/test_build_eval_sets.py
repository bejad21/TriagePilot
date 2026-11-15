import json

import pandas as pd
import pytest

import build_eval_sets as b


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _wix_fixture(tmp_path, rows, url_map, files, total=None):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in files:
        (corpus / name).write_text("x", encoding="utf-8")
    return (_write(tmp_path / "e.json", rows), _write(tmp_path / "m.json", url_map),
            corpus)


def _row(answer="a", url="u1"):
    return {"question": "q", "expected_answer": answer, "source_url": url}


def test_wix_truncation_flag_at_exactly_1000(tmp_path):
    rows = [_row("x" * 1000), _row("x" * 999)]
    e, m, c = _wix_fixture(tmp_path, rows, {"u1": "a.txt"}, ["a.txt"])
    out = b.build_wix_v2(e, m, c, expected_rows=2)
    assert [r["answer_truncated"] for r in out] == [True, False]
    assert [r["id"] for r in out] == [1, 2]
    assert out[0]["expected_behavior"] == "answer"


def test_wix_multi_url_split_and_missing_file(tmp_path):
    rows = [_row(url="u1; u2 ;u3")]
    e, m, c = _wix_fixture(tmp_path, rows, {"u1": "a.txt", "u2": "gone.txt"}, ["a.txt"])
    (row,) = b.build_wix_v2(e, m, c, expected_rows=1)
    assert row["gold_urls"] == ["u1", "u2", "u3"]
    assert row["gold_sources"] == ["a.txt"]
    assert row["unmapped_urls"] == ["u2", "u3"]


def test_wix_unmapped_url_leaves_gold_sources_empty(tmp_path):
    e, m, c = _wix_fixture(tmp_path, [_row(url="nope")], {}, [])
    (row,) = b.build_wix_v2(e, m, c, expected_rows=1)
    assert row["gold_sources"] == []
    assert row["unmapped_urls"] == ["nope"]


def test_wix_row_count_assertion(tmp_path):
    e, m, c = _wix_fixture(tmp_path, [_row()], {}, [])
    with pytest.raises(AssertionError):
        b.build_wix_v2(e, m, c)


def test_techqa_duplicates_truncation_and_gold(tmp_path):
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "swg1.txt").write_text("x", encoding="utf-8")
    rows = [
        {"question": "same", "expected_answer": "x" * 1000, "source_url": "swg1.txt"},
        {"question": "other", "expected_answer": "y", "source_url": "missing.txt"},
        {"question": "same", "expected_answer": "z", "source_url": "swg1.txt"},
    ]
    out = b.build_techqa_v2(_write(tmp_path / "t.json", rows), corpus, expected_rows=3)
    assert [r["duplicate_of"] for r in out] == [None, None, 1]
    assert [r["answer_truncated"] for r in out] == [True, False, False]
    assert [r["gold_sources"] for r in out] == [["swg1.txt"], [], ["swg1.txt"]]
    assert out[0]["expected_behavior"] == "answer"
    assert out[0]["origin"] == "original"


def test_wix_rows_have_origin_original(tmp_path):
    e, m, c = _wix_fixture(tmp_path, [_row()], {"u1": "a.txt"}, ["a.txt"])
    assert b.build_wix_v2(e, m, c, expected_rows=1)[0]["origin"] == "original"


def test_unanswerable_filter_and_order():
    df = pd.DataFrame({
        "question": ["q1", "q2", "q3", "q4"],
        "is_impossible": [True, False, True, True],
    })
    out = b.build_techqa_unanswerable(df, {"q3"})
    assert out == [{"id": 1, "question": "q1", "gold_sources": [],
                    "expected_behavior": "escalate", "origin": "published"},
                   {"id": 2, "question": "q4", "gold_sources": [],
                    "expected_behavior": "escalate", "origin": "published"}]


def test_write_json_refuses_existing(tmp_path):
    p = tmp_path / "out.json"
    b.write_json(p, [{"a": 1}])
    assert json.loads(p.read_text(encoding="utf-8")) == [{"a": 1}]
    with pytest.raises(FileExistsError):
        b.write_json(p, [])
    assert json.loads(p.read_text(encoding="utf-8")) == [{"a": 1}]


def test_cli_techqa_unanswerable_uses_fetch_and_reports_counts(tmp_path, monkeypatch, capsys):
    df = pd.DataFrame({"question": ["a", "b"], "is_impossible": [True, True]})
    seen = []
    monkeypatch.setattr(b, "fetch_techqa_parquet", lambda url: seen.append(url) or df)
    monkeypatch.setattr(b, "DATA_DIR", tmp_path)
    monkeypatch.setattr(b, "TECHQA_EVAL", _write(tmp_path / "t.json", [
        {"question": "b", "expected_answer": "", "source_url": ""}]))
    assert b.main(["techqa-unanswerable"]) == 0
    assert seen == [b.TECHQA_URL]
    assert "total rows: 2" in capsys.readouterr().out
    assert b.main(["techqa-unanswerable"]) == 1


def test_unanswerable_overlap_compares_first_1000_chars():
    long_q = "x" * 1200
    df = pd.DataFrame({"question": [long_q], "is_impossible": [True]})
    assert b.build_techqa_unanswerable(df, {long_q[:1000]}) == []


def test_unanswerable_long_question_differing_early_is_kept():
    answerable = ("a" + "x" * 1199)[:1000]
    df = pd.DataFrame({"question": ["b" + "x" * 1199], "is_impossible": [True]})
    assert len(b.build_techqa_unanswerable(df, {answerable})) == 1

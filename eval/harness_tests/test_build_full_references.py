import copy

import pandas as pd
import pytest

import build_full_references as f

CUT = 1000


def _row(rid, question="q", answer="a"):
    return {"id": rid, "question": question, "expected_answer": answer,
            "answer_truncated": len(answer) == CUT, "origin": "original"}


def _upstream(*pairs):
    return pd.DataFrame({"question": [q for q, _ in pairs],
                         "answer": [a for _, a in pairs]})


def test_cut_answer_replaced_and_flagged():
    full = "x" * 1500
    out = f.merge_full_text([_row(1, "q", full[:CUT])], _upstream(("q", full)))
    assert out[0]["expected_answer"] == full
    assert out[0]["answer_was_truncated_in_v2"] is True
    assert out[0]["answer_truncated"] is False


def test_uncut_answer_unchanged_and_not_flagged():
    out = f.merge_full_text([_row(1, "q", "short")], _upstream(("q", "short")))
    assert out[0]["expected_answer"] == "short"
    assert out[0]["answer_was_truncated_in_v2"] is False
    assert out[0]["answer_truncated"] is False
    assert out[0]["origin"] == "original"


def test_cut_question_matched_and_replaced():
    full_q = "w" * 1300
    out = f.merge_full_text([_row(1, full_q[:CUT], "a")], _upstream((full_q, "a")))
    assert out[0]["question"] == full_q


def test_same_question_disambiguated_by_answer_prefix():
    full = "y" * 1200
    up = _upstream(("q", "other"), ("q", full))
    out = f.merge_full_text([_row(1, "q", full[:CUT])], up)
    assert out[0]["expected_answer"] == full


def test_still_ambiguous_raises_with_id():
    prefix = "s" * CUT
    up = _upstream(("q", prefix + "a"), ("q", prefix + "b"))
    with pytest.raises(ValueError, match="7"):
        f.merge_full_text([_row(7, "q", prefix)], up)


def test_unmatched_raises_with_id():
    with pytest.raises(ValueError, match="9"):
        f.merge_full_text([_row(9, "missing", "a")], _upstream(("q", "a")))


def test_same_question_no_answer_match_raises():
    up = _upstream(("q", "one"), ("q", "two"))
    with pytest.raises(ValueError, match="3"):
        f.merge_full_text([_row(3, "q", "zzz")], up)


def test_counts_preserved_and_inputs_not_mutated():
    rows = [_row(1, "q1", "a1"), _row(2, "q2", "a2")]
    before = copy.deepcopy(rows)
    up = _upstream(("q1", "a1"), ("q2", "a2"))
    up_before = up.copy()
    out = f.merge_full_text(rows, up)
    assert len(out) == 2
    assert rows == before
    pd.testing.assert_frame_equal(up, up_before)
    assert "answer_was_truncated_in_v2" not in rows[0]


def test_custom_columns():
    up = pd.DataFrame({"qq": ["q"], "aa": ["full"]})
    out = f.merge_full_text([_row(1, "q", "fu")], up, question_col="qq", answer_col="aa")
    assert out[0]["expected_answer"] == "full"


def test_failed_write_removes_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "out.json"

    def boom(p, rows):
        open(p, "x").close()
        raise OSError("disk")

    monkeypatch.setattr(f, "write_json", boom)
    with pytest.raises(OSError):
        f._write_new(str(path), [])
    assert not path.exists()


def test_existing_file_survives_write_attempt(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        f._write_new(str(path), [])
    assert path.read_text(encoding="utf-8") == "keep"


def test_verbatim_duplicate_upstream_rows_are_not_ambiguous():
    up = _upstream(("q", "full"), ("q", "full"))
    out = f.merge_full_text([_row(1, "q", "full")], up)
    assert out[0]["expected_answer"] == "full"

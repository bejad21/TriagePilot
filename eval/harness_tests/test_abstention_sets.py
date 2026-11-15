import json

import pytest

from harness import abstention_sets
from harness.abstention_sets import cross_domain_rows, heldout_rows


def make_rows(n=60):
    return [
        {
            "id": i,
            "question": f"q{i}",
            "expected_answer": f"a{i}",
            "gold_sources": [f"s{i}.txt"],
            "expected_behavior": "answer",
            "origin": "original",
        }
        for i in range(1, n + 1)
    ]


def test_deterministic_for_same_seed():
    rows = make_rows()
    assert cross_domain_rows(rows, "wix", 10, seed=3) == cross_domain_rows(rows, "wix", 10, seed=3)


def test_file_order_does_not_change_result():
    rows = make_rows()
    assert cross_domain_rows(rows, "wix", 10) == cross_domain_rows(rows[::-1], "wix", 10)


def test_different_seed_differs():
    rows = make_rows()
    a = cross_domain_rows(rows, "wix", 10, seed=0)
    b = cross_domain_rows(rows, "wix", 10, seed=1)
    assert [r["source_id"] for r in a] != [r["source_id"] for r in b]


def test_relabels_rows_and_keeps_question():
    out = cross_domain_rows(make_rows(), "techqa", 5)
    assert [r["id"] for r in out] == [1, 2, 3, 4, 5]
    for r in out:
        assert r == {
            "id": r["id"],
            "question": f"q{r['source_id']}",
            "gold_sources": [],
            "expected_behavior": "escalate",
            "origin": "mechanical",
            "source_dataset": "techqa",
            "source_id": r["source_id"],
        }
    assert len({r["source_id"] for r in out}) == 5


def test_exclude_ids_and_non_answer_rows():
    rows = make_rows(20)
    rows[19]["expected_behavior"] = "escalate"
    out = cross_domain_rows(rows, "wix", 9, exclude_ids=range(1, 11))
    assert {r["source_id"] for r in out} == set(range(11, 20))


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        cross_domain_rows(make_rows(10), "wix", 5, exclude_ids=range(1, 8))


def test_heldout_range_inclusive_and_filters():
    rows = make_rows(140)
    rows[39]["duplicate_of"] = 5
    rows[40]["duplicate_of"] = None
    rows[49]["expected_behavior"] = "escalate"
    ids = [r["id"] for r in heldout_rows(rows)]
    assert ids[0] == 31 and ids[-1] == 130
    assert 40 not in ids and 50 not in ids and 41 in ids
    assert len(ids) == 98


def test_assert_not_tuned_names_offending_ids():
    rows = make_rows(40)
    with pytest.raises(ValueError, match=r"\[1, 2, 30\]"):
        abstention_sets.assert_not_tuned([rows[1], rows[0], rows[29], rows[30]])
    abstention_sets.assert_not_tuned(rows[30:])
    abstention_sets.assert_not_tuned(heldout_rows(make_rows(140)))
    abstention_sets.assert_not_tuned(rows[5:], tuned_max=5)
    with pytest.raises(ValueError):
        abstention_sets.assert_not_tuned(rows[30:], tuned_max=31)


def test_main_refuses_to_overwrite(tmp_path):
    (tmp_path / "wix_full.json").write_text(json.dumps(make_rows(60)))
    (tmp_path / "techqa_full.json").write_text(json.dumps(make_rows(60)))
    (tmp_path / "wix_to_techqa_cross.json").write_text("[]")
    with pytest.raises(FileExistsError, match="wix_to_techqa_cross.json"):
        abstention_sets.write_cross_sets(tmp_path, n=10)
    assert not (tmp_path / "techqa_to_wix_cross.json").exists()


def test_write_cross_sets_creates_files(tmp_path):
    (tmp_path / "wix_full.json").write_text(json.dumps(make_rows(60)))
    tq = make_rows(60)
    tq[40]["duplicate_of"] = 3
    (tmp_path / "techqa_full.json").write_text(json.dumps(tq))
    abstention_sets.write_cross_sets(tmp_path, n=10)
    w = json.loads((tmp_path / "wix_to_techqa_cross.json").read_text())
    t = json.loads((tmp_path / "techqa_to_wix_cross.json").read_text())
    assert len(w) == len(t) == 10
    assert all(r["source_id"] > 30 for r in w + t)
    assert 41 not in {r["source_id"] for r in t}
    assert {r["source_dataset"] for r in w} == {"wix"}
    assert {r["source_dataset"] for r in t} == {"techqa"}


def test_write_cross_sets_reads_full_length_questions(tmp_path):
    for name in ("wix", "techqa"):
        full = make_rows(60)
        for r in full:
            r["question"] = f"full {r['id']} " + "x" * 1200
        (tmp_path / f"{name}_full.json").write_text(json.dumps(full))
        (tmp_path / f"{name}_v2.json").write_text(json.dumps(make_rows(60)))
    abstention_sets.write_cross_sets(tmp_path, n=10)
    for out in ("wix_to_techqa_cross.json", "techqa_to_wix_cross.json"):
        for r in json.loads((tmp_path / out).read_text()):
            assert r["question"] == f"full {r['source_id']} " + "x" * 1200


def test_compute_failure_leaves_no_output(tmp_path):
    (tmp_path / "wix_full.json").write_text(json.dumps(make_rows(60)))
    with pytest.raises(FileNotFoundError):
        abstention_sets.write_cross_sets(tmp_path, n=10)
    assert not (tmp_path / "wix_to_techqa_cross.json").exists()


def test_too_few_rows_in_second_source_leaves_no_output(tmp_path):
    (tmp_path / "wix_full.json").write_text(json.dumps(make_rows(60)))
    (tmp_path / "techqa_full.json").write_text(json.dumps(make_rows(35)))
    with pytest.raises(ValueError):
        abstention_sets.write_cross_sets(tmp_path, n=10)
    assert not (tmp_path / "wix_to_techqa_cross.json").exists()


def test_second_write_failure_removes_only_first_file(tmp_path, monkeypatch):
    (tmp_path / "wix_full.json").write_text(json.dumps(make_rows(60)))
    (tmp_path / "techqa_full.json").write_text(json.dumps(make_rows(60)))
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("keep")
    real = abstention_sets.write_new_json

    def flaky(path, rows, created):
        if path.name == "techqa_to_wix_cross.json":
            raise OSError("disk full")
        real(path, rows, created)

    monkeypatch.setattr(abstention_sets, "write_new_json", flaky)
    with pytest.raises(OSError, match="disk full"):
        abstention_sets.write_cross_sets(tmp_path, n=10)
    assert not (tmp_path / "wix_to_techqa_cross.json").exists()
    assert unrelated.read_text() == "keep"


def setup_sources(tmp_path):
    (tmp_path / "wix_full.json").write_text(json.dumps(make_rows(60)))
    (tmp_path / "techqa_full.json").write_text(json.dumps(make_rows(60)))


def test_midway_write_failure_leaves_no_file(tmp_path, monkeypatch):
    setup_sources(tmp_path)

    def boom(*args, **kwargs):
        raise TypeError("cannot encode")

    monkeypatch.setattr(abstention_sets.json, "dumps", boom)
    with pytest.raises(TypeError, match="cannot encode"):
        abstention_sets.write_cross_sets(tmp_path, n=10)
    assert not (tmp_path / "wix_to_techqa_cross.json").exists()
    assert not (tmp_path / "techqa_to_wix_cross.json").exists()


def test_file_created_by_someone_else_is_not_deleted(tmp_path, monkeypatch):
    setup_sources(tmp_path)
    real = abstention_sets.write_new_json
    other = tmp_path / "techqa_to_wix_cross.json"

    def racing(path, rows, created):
        real(path, rows, created)
        other.write_text("theirs")

    monkeypatch.setattr(abstention_sets, "write_new_json", racing)
    with pytest.raises(FileExistsError):
        abstention_sets.write_cross_sets(tmp_path, n=10)
    assert other.read_text() == "theirs"
    assert not (tmp_path / "wix_to_techqa_cross.json").exists()


def test_cleanup_error_does_not_mask_original(tmp_path, monkeypatch):
    setup_sources(tmp_path)
    real = abstention_sets.write_new_json

    def flaky(path, rows, created):
        if path.name == "techqa_to_wix_cross.json":
            raise RuntimeError("original")
        real(path, rows, created)

    def bad_unlink(self, missing_ok=False):
        raise PermissionError("locked")

    monkeypatch.setattr(abstention_sets, "write_new_json", flaky)
    monkeypatch.setattr(abstention_sets.Path, "unlink", bad_unlink)
    with pytest.raises(RuntimeError, match="original"):
        abstention_sets.write_cross_sets(tmp_path, n=10)

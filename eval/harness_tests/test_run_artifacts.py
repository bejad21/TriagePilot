import json

import pytest

from harness.run_artifacts import RunRecord, RunStore, validate_record


def make(row_id: int, **kw) -> RunRecord:
    base = dict(row_id=row_id, dataset="d", split="test", question="q?",
                expected_behavior="answer", status="answered")
    base.update(kw)
    return RunRecord(**base)


def test_round_trip_preserves_fields(tmp_path):
    store = RunStore(tmp_path / "run.jsonl")
    rec = make(1, answer="café", retrieved=[{"source": "s", "text": "t", "score": None}],
               config={"k": 5}, gold_sources=["s"], duration_s=1.5)
    store.append(rec)
    assert store.load() == [rec]
    assert "café" in (tmp_path / "run.jsonl").read_text(encoding="utf-8")


def test_context_collected_round_trips_and_old_lines_default_false(tmp_path):
    store = RunStore(tmp_path / "run.jsonl")
    store.append(make(1, context_collected=True))
    old = {"row_id": 2, "dataset": "d", "split": "test", "question": "q?",
           "expected_behavior": "answer", "status": "answered"}
    with open(store.path, "a", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")
    assert [r.context_collected for r in store.load()] == [True, False]


def test_done_ids_and_iter(tmp_path):
    store = RunStore(tmp_path / "run.jsonl")
    assert store.done_ids() == set()
    store.append(make(1))
    store.append(make(2))
    assert store.done_ids() == {1, 2}
    assert [r.row_id for r in store] == [1, 2]


def test_creates_parent_folders(tmp_path):
    path = tmp_path / "a" / "b" / "run.jsonl"
    RunStore(path).append(make(1))
    assert path.exists()


def test_partial_trailing_line_ignored(tmp_path):
    path = tmp_path / "run.jsonl"
    store = RunStore(path)
    store.append(make(1))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"row_id": 2, "data')
    assert store.done_ids() == {1}
    assert [r.row_id for r in store.load()] == [1]


def test_append_after_partial_line_keeps_store_valid(tmp_path):
    path = tmp_path / "run.jsonl"
    store = RunStore(path)
    store.append(make(1))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"row_id": 2, "data')
    store.append(make(2))
    assert [r.row_id for r in store.load()] == [1, 2]


def test_append_after_complete_line_missing_newline(tmp_path):
    path = tmp_path / "run.jsonl"
    store = RunStore(path)
    store.append(make(1))
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    store.append(make(2))
    assert [r.row_id for r in store.load()] == [1, 2]


def test_invalid_middle_line_raises(tmp_path):
    path = tmp_path / "run.jsonl"
    store = RunStore(path)
    store.append(make(1))
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json\n")
    store.append(make(3))
    with pytest.raises(ValueError):
        store.load()
    with pytest.raises(ValueError):
        store.done_ids()


def test_middle_line_with_wrong_fields_raises(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"bogus": 1}) + "\n" + json.dumps({"bogus": 2}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError):
        RunStore(path).load()


def test_duplicate_row_id_last_wins(tmp_path):
    store = RunStore(tmp_path / "run.jsonl")
    store.append(make(1, status="error", error="boom"))
    store.append(make(2))
    store.append(make(1, status="answered", answer="ok"))
    rows = store.load()
    assert [r.row_id for r in rows] == [1, 2]
    assert rows[0].status == "answered" and rows[0].answer == "ok"
    assert store.done_ids() == {1, 2}


def test_missing_file_is_empty(tmp_path):
    store = RunStore(tmp_path / "none.jsonl")
    assert store.load() == []
    assert store.done_ids() == set()


@pytest.mark.parametrize("kw", [
    {"status": "bogus"},
    {"status": ""},
    {"expected_behavior": "maybe"},
    {"question": ""},
])
def test_validate_record_rejects(kw):
    with pytest.raises(ValueError):
        validate_record(make(1, **kw))


@pytest.mark.parametrize("status", ["answered", "escalated", "rejected", "error"])
def test_validate_record_accepts(status):
    validate_record(make(1, status=status))
    validate_record(make(1, expected_behavior="escalate"))


def test_truncated_multibyte_tail_ignored_and_repaired(tmp_path):
    path = tmp_path / "run.jsonl"
    store = RunStore(path)
    store.append(make(1, answer="caf\u00e9"))
    with open(path, "ab") as f:
        f.write(b'{"row_id": 2, "answer": "' + "\u00e9".encode("utf-8")[:1])
    assert store.done_ids() == {1}
    assert [r.row_id for r in store.load()] == [1]
    store.append(make(2))
    assert b"\xc3" not in path.read_bytes().replace("\u00e9".encode("utf-8"), b"")
    assert [r.row_id for r in store.load()] == [1, 2]


def test_invalid_utf8_middle_line_raises(tmp_path):
    path = tmp_path / "run.jsonl"
    store = RunStore(path)
    store.append(make(1))
    with open(path, "ab") as f:
        f.write(b'{"row_id": 2, "answer": "\xc3"}\n')
    store.append(make(3))
    with pytest.raises(ValueError):
        store.load()

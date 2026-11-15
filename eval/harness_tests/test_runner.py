import json

import pytest
import requests

from harness.run_artifacts import RunStore
from harness.runner import load_rows_json, run_rows

URL = "http://webhook.invalid/triage"
ANSWERED = {"status": "answered", "answer": "a", "cited_source": "s.md"}
ESCALATED = {"status": "escalated", "message": "human needed"}
REJECTED = {"status": "rejected", "reason": "off topic"}


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Plays back scripted responses (or exceptions) and records every POST."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rows(n=3):
    return [{"id": i, "question": f"q{i}?"} for i in range(1, n + 1)]


def run(tmp_path, session, rws=None, **kw):
    store = RunStore(tmp_path / "run.jsonl")
    kw.setdefault("config", {"k": 5})
    counts = run_rows(rws if rws is not None else rows(), URL, store, dataset="d",
                      split="test", session=session, sleep=lambda s: None, **kw)
    return store, counts


def test_three_rows_three_statuses(tmp_path):
    session = FakeSession([FakeResponse(ANSWERED), FakeResponse(ESCALATED), FakeResponse(REJECTED)])
    store, counts = run(tmp_path, session)
    recs = store.load()
    assert [r.status for r in recs] == ["answered", "escalated", "rejected"]
    assert recs[0].answer == "a" and recs[0].cited_source == "s.md"
    assert recs[1].reason == "human needed"
    assert recs[2].reason == "off topic"
    assert counts == {"answered": 1, "escalated": 1, "rejected": 1, "error": 0}
    assert recs[0].config == {"k": 5}
    assert recs[0].retrieved == [] and recs[0].confidence is None
    assert recs[0].expected_behavior == "answer" and recs[0].gold_sources == []
    assert recs[0].started_at.endswith("+00:00") and recs[0].duration_s >= 0


def test_request_body_is_exactly_text_and_one_post_per_row(tmp_path):
    session = FakeSession([FakeResponse(ANSWERED)] * 2)
    run(tmp_path, session, rows(2), timeout=42.0)
    assert [c["json"] for c in session.calls] == [{"text": "q1?"}, {"text": "q2?"}]
    assert session.calls[0]["url"] == URL and session.calls[0]["timeout"] == 42.0


def test_optional_row_keys_are_stored(tmp_path):
    row = {"id": 7, "question": "q", "expected_answer": "x", "gold_sources": ["g"],
           "expected_behavior": "escalate"}
    store, _ = run(tmp_path, FakeSession([FakeResponse(ESCALATED)]), [row])
    rec = store.load()[0]
    assert (rec.expected_answer, rec.gold_sources, rec.expected_behavior) == ("x", ["g"], "escalate")


def test_rerun_skips_finished_rows(tmp_path):
    run(tmp_path, FakeSession([FakeResponse(ANSWERED)] * 3))
    session = FakeSession([])
    _, counts = run(tmp_path, session)
    assert session.calls == []
    assert sum(counts.values()) == 0


def test_malformed_body_becomes_error_and_loop_continues(tmp_path):
    session = FakeSession([FakeResponse("<html>oops"), FakeResponse(ANSWERED)])
    store, counts = run(tmp_path, session, rows(2))
    first, second = store.load()
    assert first.status == "error" and first.error
    assert second.status == "answered"
    assert counts["error"] == 1


@pytest.mark.parametrize("body", [
    "[1, 2]", "null", {"status": "weird"}, {"answer": "no status"},
    {"status": "answered"}, {"status": "rejected"}])
def test_bad_shapes_become_error(tmp_path, body):
    store, _ = run(tmp_path, FakeSession([FakeResponse(body)]), rows(1))
    assert store.load()[0].status == "error"


def test_http_error_is_error_with_truncated_body_and_not_retried(tmp_path):
    session = FakeSession([FakeResponse("x" * 5000, 500), FakeResponse(ANSWERED)])
    store, _ = run(tmp_path, session, rows(2))
    first = store.load()[0]
    assert first.status == "error" and "500" in first.error
    assert len(first.error) < 400
    assert len(session.calls) == 2


def test_timeout_is_error_and_never_retried(tmp_path):
    session = FakeSession([requests.exceptions.ReadTimeout("slow"), FakeResponse(ANSWERED)])
    store, _ = run(tmp_path, session, rows(2))
    assert [r.status for r in store.load()] == ["error", "answered"]
    assert len(session.calls) == 2


def test_connect_timeout_is_not_retried(tmp_path):
    session = FakeSession([requests.exceptions.ConnectTimeout("slow")])
    store, _ = run(tmp_path, session, rows(1))
    assert store.load()[0].status == "error" and len(session.calls) == 1


def refused():
    err = requests.exceptions.ConnectionError("refused")
    err.__cause__ = ConnectionRefusedError(10061, "refused")
    return err


def test_connection_refused_retried_once(tmp_path):
    session = FakeSession([refused(), FakeResponse(ANSWERED)])
    store, _ = run(tmp_path, session, rows(1))
    assert store.load()[0].status == "answered" and len(session.calls) == 2


def test_connection_refused_twice_is_error(tmp_path):
    session = FakeSession([refused(), refused()])
    store, _ = run(tmp_path, session, rows(1))
    assert store.load()[0].status == "error" and len(session.calls) == 2


def test_other_connection_error_not_retried(tmp_path):
    session = FakeSession([requests.exceptions.ConnectionError("reset")])
    store, _ = run(tmp_path, session, rows(1))
    assert store.load()[0].status == "error" and len(session.calls) == 1


def test_stop_file_halts_before_next_row(tmp_path):
    class StopAfterFirst(FakeSession):
        def post(self, url, json=None, timeout=None):
            (tmp_path / "STOP").write_text("")
            return super().post(url, json=json, timeout=timeout)

    session = StopAfterFirst([FakeResponse(ANSWERED)] * 3)
    store, _ = run(tmp_path, session)
    assert [r.row_id for r in store.load()] == [1]
    assert len(session.calls) == 1


def test_stop_file_present_at_start_runs_nothing(tmp_path):
    (tmp_path / "STOP").write_text("")
    session = FakeSession([])
    run(tmp_path, session)
    assert session.calls == []


def test_error_rows_are_done_unless_retry_errors(tmp_path):
    run(tmp_path, FakeSession([FakeResponse("bad"), FakeResponse(ANSWERED)]), rows(2))
    session = FakeSession([])
    run(tmp_path, session, rows(2))
    assert session.calls == []
    session = FakeSession([FakeResponse(ANSWERED)])
    store, _ = run(tmp_path, session, rows(2), retry_errors=True)
    assert [c["json"] for c in session.calls] == [{"text": "q1?"}]
    assert [r.status for r in store.load()] == ["answered", "answered"]
    session = FakeSession([])
    run(tmp_path, session, rows(2), retry_errors=True)
    assert session.calls == []


def test_pause_between_rows(tmp_path):
    pauses = []
    store = RunStore(tmp_path / "run.jsonl")
    run_rows(rows(2), URL, store, dataset="d", split="t", config={}, pause_s=2.5,
             session=FakeSession([FakeResponse(ANSWERED)] * 2), sleep=pauses.append)
    assert pauses == [2.5]


def test_injected_clocks(tmp_path):
    ticks = iter([10.0, 13.5])
    store = RunStore(tmp_path / "run.jsonl")
    run_rows(rows(1), URL, store, dataset="d", split="t", config={},
             session=FakeSession([FakeResponse(ANSWERED)]), sleep=lambda s: None,
             now=lambda: "2026-01-01T00:00:00+00:00", clock=lambda: next(ticks))
    rec = store.load()[0]
    assert rec.started_at == "2026-01-01T00:00:00+00:00" and rec.duration_s == 3.5


def test_invalid_row_behavior_raises_before_any_call(tmp_path):
    session = FakeSession([])
    with pytest.raises(ValueError):
        run(tmp_path, session, [{"id": 1, "question": "q", "expected_behavior": "bogus"}])
    assert session.calls == []


def test_mixing_data_files_in_one_store_raises_before_any_post(tmp_path):
    store = RunStore(tmp_path / "run.jsonl")
    heldout = [{"id": i, "question": f"h{i}?"} for i in range(31, 43)]
    run_rows(heldout, URL, store, dataset="wix", split="heldout", config={},
             session=FakeSession([FakeResponse(ANSWERED)] * 12), sleep=lambda s: None)
    cross = [{"id": i, "question": f"c{i}?", "expected_behavior": "escalate"}
             for i in range(1, 41)]
    for dataset, split in (("wix", "cross"), ("techqa_to_wix", "heldout")):
        session = FakeSession([])
        with pytest.raises(ValueError, match=r"one run store per \(dataset, split\)"):
            run_rows(cross, URL, store, dataset=dataset, split=split, config={},
                     session=session, sleep=lambda s: None)
        assert session.calls == []
    assert len(store.load()) == 12


def test_duplicate_row_ids_raise_before_any_post(tmp_path):
    session = FakeSession([])
    with pytest.raises(ValueError, match="duplicate"):
        run(tmp_path, session, [{"id": 1, "question": "a"}, {"id": 1, "question": "b"}])
    assert session.calls == []


def test_resume_with_same_dataset_and_split(tmp_path):
    run(tmp_path, FakeSession([FakeResponse(ANSWERED)]), rows(1))
    session = FakeSession([FakeResponse(ANSWERED)] * 2)
    store, counts = run(tmp_path, session, rows(3))
    assert [c["json"] for c in session.calls] == [{"text": "q2?"}, {"text": "q3?"}]
    assert [r.row_id for r in store.load()] == [1, 2, 3]


def test_load_rows_json(tmp_path):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps([{"id": 1, "question": "q"}]), encoding="utf-8")
    assert load_rows_json(p) == [{"id": 1, "question": "q"}]


@pytest.mark.parametrize("data", [{"id": 1}, [1], [{"id": 1}], [{"question": "q"}]])
def test_load_rows_json_rejects_bad_shapes(tmp_path, data):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rows_json(p)


def test_new_session_sends_the_webhook_secret_only_when_set(monkeypatch):
    from harness.runner import new_session
    monkeypatch.delenv("TRIAGEPILOT_WEBHOOK_KEY", raising=False)
    assert "X-TriagePilot-Key" not in new_session().headers
    monkeypatch.setenv("TRIAGEPILOT_WEBHOOK_KEY", "s3cret")
    assert new_session().headers["X-TriagePilot-Key"] == "s3cret"

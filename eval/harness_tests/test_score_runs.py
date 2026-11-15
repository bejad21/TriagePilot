import json

import pytest

from harness import report, score_runs
from harness.backends import BackendError
from harness.metrics import ScoredRow
from harness.run_artifacts import RunRecord


class ScriptedChat:
    """Replies by prompt kind so call order does not matter."""
    name = "fake"

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.calls = []

    def complete(self, system, user, *, temperature=0.0):
        self.calls.append(user)
        if self.fail_on and self.fail_on in user:
            raise BackendError("boom")
        if "Chunk 1" in user:
            return json.dumps({"relevance": [True, False]})
        if '"attributed"' in user:
            return json.dumps({"claims": [{"claim": "a", "attributed": True}]})
        if '"supported"' in user:
            return json.dumps({"claims": [{"claim": "a", "supported": True},
                                          {"claim": "b", "supported": False}]})
        if '"score"' in user:
            return json.dumps({"score": 4})
        return json.dumps({"questions": ["q1"]})


class FakeEmbedder:
    name = "e"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def rec(row_id, status, behavior="answer", **kw):
    base = dict(row_id=row_id, dataset="d", split="real", question=f"question {row_id}",
                expected_behavior=behavior, expected_answer="ref", gold_sources=["g"],
                status=status, answer="ans" if status == "answered" else None,
                retrieved=[{"source": "x", "text": "tx"}, {"source": "g", "text": "tg"}])
    base.update(kw)
    return RunRecord(**base)


def test_three_record_run():
    records = [rec(1, "answered"), rec(2, "escalated"),
               rec(3, "answered", behavior="escalate")]
    chat = ScriptedChat()
    rows = score_runs.score_records(records, chat, FakeEmbedder())
    assert [r.row_id for r in rows] == [1, 2]
    answered, escalated = rows
    assert answered.answered
    assert answered.faithfulness == 0.5
    assert answered.answer_relevancy == 1.0
    assert answered.answer_correctness == 4
    assert answered.context_precision == 1.0
    assert answered.context_recall == 1.0
    assert answered.hit_at_k == 1.0
    assert answered.reciprocal_rank == 0.5
    assert not escalated.answered
    assert escalated.faithfulness is None and escalated.answer_correctness is None
    assert escalated.context_precision == 1.0 and escalated.reciprocal_rank == 0.5


def test_no_gold_sources_gives_none_retrieval_metrics():
    rows = score_runs.score_records([rec(1, "answered", gold_sources=[])],
                                    ScriptedChat(), FakeEmbedder())
    assert rows[0].hit_at_k is None and rows[0].reciprocal_rank is None


def test_error_and_rejected_make_no_judge_calls():
    chat = ScriptedChat()
    rows = score_runs.score_records([rec(1, "error"), rec(2, "rejected")],
                                    chat, FakeEmbedder())
    assert chat.calls == []
    assert all(not r.answered and r.hit_at_k is None and r.context_precision is None
               for r in rows)


def test_run_errors_are_excluded_from_rows_and_listed():
    chat = ScriptedChat()
    rows, errors = score_runs.score_records_with_errors(
        [rec(1, "error", error="ReadTimeout: slow"), rec(2, "rejected")], chat, FakeEmbedder())
    assert [r.row_id for r in rows] == [2]
    assert errors == [(1, "run", "ReadTimeout: slow")]
    assert chat.calls == []


def test_judge_error_leaves_metric_none_and_is_reported():
    chat = ScriptedChat(fail_on="Chunk 1")
    rows, errors = score_runs.score_records_with_errors(
        [rec(1, "answered")], chat, FakeEmbedder())
    assert rows[0].context_precision is None
    assert rows[0].faithfulness == 0.5
    assert len(errors) == 1
    assert errors[0][:2] == (1, "context_precision") and "boom" in errors[0][2]


CONTEXT_METRICS = ("faithfulness", "context_precision", "context_recall",
                   "hit_at_k", "reciprocal_rank")


def test_uncollected_context_is_excluded_not_a_miss():
    chat = ScriptedChat()
    rows = score_runs.score_records([rec(1, "answered", retrieved=[])], chat, FakeEmbedder())
    row = rows[0]
    assert all(getattr(row, m) is None for m in CONTEXT_METRICS)
    assert row.answer_relevancy == 1.0 and row.answer_correctness == 4
    table = report.metric_table(rows)
    hit_line = next(ln for ln in table.splitlines() if ln.startswith("| hit_at_k "))
    assert "1 without a value excluded" in hit_line and "0.000" not in hit_line


def test_collected_empty_context_is_a_real_miss():
    rows = score_runs.score_records(
        [rec(1, "answered", retrieved=[], context_collected=True)],
        ScriptedChat(), FakeEmbedder())
    assert rows[0].hit_at_k == 0.0 and rows[0].reciprocal_rank == 0.0
    assert rows[0].faithfulness == 0.5


def test_retrieval_metrics_use_only_the_first_ctx_k_chunks():
    retrieved = [{"source": f"s{i}", "text": f"t{i}"} for i in range(1, 11)]
    record = rec(1, "escalated", gold_sources=["s8"], retrieved=retrieved)
    at6 = score_runs.score_records([record], ScriptedChat(), FakeEmbedder())[0]
    assert at6.hit_at_k == 0.0 and at6.reciprocal_rank == 0.0
    at10 = score_runs.score_records([record], ScriptedChat(), FakeEmbedder(), ctx_k=10)[0]
    assert at10.hit_at_k == 1.0 and at10.reciprocal_rank == 1 / 8


def test_metric_table_states_k():
    rows = [ScoredRow(1, True, hit_at_k=1.0)]
    assert "K = first 6 chunks ranked by reranker score" in report.metric_table(rows, ctx_k=6)
    assert "K = first ctx_k chunks" in report.metric_table(rows)
    assert "K = " not in report.metric_table(rows, k_note=False)


ROWS = [ScoredRow(1, True, 0.5, None, 4.0, 0.25, 1.0, 1.0, 0.5), ScoredRow(2, False)]
META = {"judge": "fake", "embedder": "e", "ctx_k": 6, "runs": "runs.jsonl",
        "created_utc": "2026-10-26T00:00:00+00:00", "judge_errors": []}


def test_scores_csv_roundtrip(tmp_path):
    path = tmp_path / "s.csv"
    score_runs.write_scores(path, ROWS, META)
    assert score_runs.read_scores(path) == ROWS
    assert score_runs.read_meta(path) == META
    assert (tmp_path / "s.csv.meta.json").exists()


def test_read_meta_without_sidecar_is_none(tmp_path):
    assert score_runs.read_meta(tmp_path / "s.csv") is None


def test_scores_meta_records_judge_errors_but_not_run_errors():
    records = [rec(1, "answered"), rec(2, "error", error="HTTP 500: x")]
    rows, errors = score_runs.score_records_with_errors(
        records, ScriptedChat(fail_on="Chunk 1"), FakeEmbedder())
    meta = score_runs.scores_meta(ScriptedChat(), FakeEmbedder(), ctx_k=6,
                                  runs_path="r.jsonl", errors=errors)
    assert meta["judge"] == "fake" and meta["embedder"] == "e" and meta["ctx_k"] == 6
    assert meta["runs"] == "r.jsonl" and meta["created_utc"].endswith("+00:00")
    assert meta["judge_errors"] == [
        {"row_id": 1, "metric": "context_precision", "message": "boom"}]


@pytest.mark.parametrize("existing", ["s.csv", "s.csv.meta.json"])
def test_write_scores_refuses_existing_files_and_writes_nothing(tmp_path, existing):
    (tmp_path / existing).write_text("theirs")
    with pytest.raises(FileExistsError):
        score_runs.write_scores(tmp_path / "s.csv", ROWS, META)
    assert sorted(p.name for p in tmp_path.iterdir()) == [existing]
    assert (tmp_path / existing).read_text() == "theirs"


def test_second_write_failure_removes_the_first_file(tmp_path, monkeypatch):
    real = score_runs._write_new

    def flaky(path, text, created):
        if path.name.endswith(".meta.json"):
            raise OSError("disk full")
        real(path, text, created)

    monkeypatch.setattr(score_runs, "_write_new", flaky)
    with pytest.raises(OSError, match="disk full"):
        score_runs.write_scores(tmp_path / "s.csv", ROWS, META)
    assert list(tmp_path.iterdir()) == []


def test_cleanup_error_does_not_mask_the_write_error(tmp_path, monkeypatch):
    real = score_runs._write_new

    def flaky(path, text, created):
        if path.name.endswith(".meta.json"):
            raise RuntimeError("original")
        real(path, text, created)

    def bad_unlink(self, missing_ok=False):
        raise PermissionError("locked")

    monkeypatch.setattr(score_runs, "_write_new", flaky)
    monkeypatch.setattr(score_runs.pathlib.Path, "unlink", bad_unlink)
    with pytest.raises(RuntimeError, match="original"):
        score_runs.write_scores(tmp_path / "s.csv", ROWS, META)

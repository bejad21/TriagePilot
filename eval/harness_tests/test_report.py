import math

import pytest

from harness import report
from harness.__main__ import main
from harness.metrics import ScoredRow
from harness.run_artifacts import RunRecord, RunStore
from harness.score_runs import write_scores


def rec(row_id, behavior, status):
    return RunRecord(row_id=row_id, dataset="d", split="real", question="q",
                     expected_behavior=behavior, status=status)


def test_metric_table_shows_mean_interval_and_basis():
    rows = [ScoredRow(i, True, faithfulness=v) for i, v in enumerate([0.8, 0.9, 1.0, 0.7])]
    table = report.metric_table(rows)
    assert "0.850" in table and "[" in table
    assert "| n |" in table and "basis" in table


def test_abstention_table_false_answer_rate():
    records = [rec(1, "escalate", "answered"), rec(2, "escalate", "answered"),
               rec(3, "escalate", "escalated"), rec(4, "escalate", "rejected"),
               rec(5, "answer", "answered"), rec(6, "answer", "escalated")]
    table = report.abstention_table(records)
    assert "2/4" in table and "false_answer_rate" in table and "[" in table
    assert "rejected" in table


def _twelve_with_one_timeout():
    records = [rec(i, "escalate", "answered" if i <= 2 else "escalated") for i in range(1, 6)]
    records += [rec(i, "answer", "answered" if i <= 9 else "escalated") for i in range(6, 12)]
    timeout = rec(12, "escalate", "error")
    timeout.error = "ReadTimeout: slow"
    return records, records + [timeout]


def test_run_error_is_excluded_from_abstention_denominators():
    clean, with_error = _twelve_with_one_timeout()
    table = report.abstention_table(with_error)
    assert "2/5 = 0.400" in table and "4/6 = 0.667" in table
    assert "/6 = " not in table.split("| answer |")[0]
    assert "run errors excluded: 1" in table
    assert "run errors excluded: 0" in report.abstention_table(clean)


def test_metric_table_prints_run_error_count():
    rows = [ScoredRow(1, True), ScoredRow(2, False)]
    table = report.metric_table(rows, run_errors=1)
    assert "run errors excluded: 1" in table
    assert "| answered_rate | 0.500 |" in table


def _write_run(tmp_path, records, meta=None):
    store = RunStore(tmp_path / "runs.jsonl")
    for r in records:
        store.append(r)
    scores = tmp_path / "scores.csv"
    rows = [ScoredRow(r.row_id, r.status == "answered")
            for r in records if r.expected_behavior == "answer"]
    write_scores(scores, rows, meta or {})
    if meta is None:
        (tmp_path / "scores.csv.meta.json").unlink()
    return ["report", "--runs", str(store.path), "--scores", str(scores)]


def test_cli_report_prints_provenance_from_sidecar(tmp_path, capsys):
    clean, _ = _twelve_with_one_timeout()
    errors = [{"row_id": 6, "metric": "faithfulness", "message": "quota"},
              {"row_id": 7, "metric": "context_recall", "message": "quota"}]
    meta = {"judge": "mistral-small-latest", "embedder": "mistral-embed", "ctx_k": 6,
            "runs": "runs.jsonl", "created_utc": "2026-10-26T00:00:00+00:00",
            "judge_errors": errors}
    assert main(_write_run(tmp_path, clean, meta)) == 0
    out = capsys.readouterr().out
    assert "judge mistral-small-latest" in out and "judge errors 2" in out
    assert "ctx_k 6" in out and "K = first 6 chunks" in out


def test_cli_report_without_sidecar_says_provenance_none(tmp_path, capsys):
    clean, _ = _twelve_with_one_timeout()
    assert main(_write_run(tmp_path, clean)) == 0
    out = capsys.readouterr().out
    assert "provenance: none" in out and "K = first ctx_k chunks" in out


def test_cli_report_refuses_runs_with_errors(tmp_path, capsys):
    _, with_error = _twelve_with_one_timeout()
    args = _write_run(tmp_path, with_error)
    assert main(args) == 1
    assert "error" in capsys.readouterr().err
    assert main(args + ["--allow-errors"]) == 0
    assert "run errors excluded: 1" in capsys.readouterr().out


def test_cli_report_without_errors_succeeds(tmp_path, capsys):
    clean, _ = _twelve_with_one_timeout()
    assert main(_write_run(tmp_path, clean)) == 0
    assert "run errors excluded: 0" in capsys.readouterr().out


def test_coverage_accuracy():
    rows = [ScoredRow(1, True, answer_correctness=5), ScoredRow(2, True, answer_correctness=2),
            ScoredRow(3, True), ScoredRow(4, False)]
    coverage, accuracy = report.coverage_accuracy(rows)
    assert coverage == 0.75 and accuracy == 0.5
    _, none_acc = report.coverage_accuracy([ScoredRow(1, True)])
    assert math.isnan(none_acc)


def test_judge_agreement_with_itself():
    rows = [ScoredRow(i, True, faithfulness=v) for i, v in enumerate([0.2, 0.9, 1.0, 0.5])]
    agreement = report.judge_agreement(rows, rows)
    assert agreement["faithfulness"]["pearson"] == pytest.approx(1.0)
    assert agreement["faithfulness"]["kappa"] == 1.0


def _correctness(values):
    return [ScoredRow(i, True, answer_correctness=v) for i, v in enumerate(values)]


def test_correctness_kappa_uses_scale_threshold():
    # >= 4 gives a=[T,T,T,F], b=[T,T,F,F]: observed 0.75, expected 0.5, kappa 0.5.
    result = report.judge_agreement(_correctness([5, 4, 4, 3]),
                                    _correctness([5, 4, 3, 3]))["answer_correctness"]
    assert result["threshold"] == 4.0
    assert result["kappa"] == pytest.approx(0.5)
    assert -1 <= result["kappa"] <= 1


def test_identical_correctness_scores_agree_fully():
    rows = _correctness([5, 4, 2, 3])
    result = report.judge_agreement(rows, rows)["answer_correctness"]
    assert result["kappa"] == 1.0
    assert result["pearson"] == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_constant_series_gives_nan_pearson_without_raising():
    rows = _correctness([4, 4, 4])
    result = report.judge_agreement(rows, rows)["answer_correctness"]
    assert math.isnan(result["pearson"])

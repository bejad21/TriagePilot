import pytest
import requests

from harness.__main__ import main
from harness.metrics import ScoredRow
from harness.score_runs import write_scores


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("network call")

    monkeypatch.setattr(requests.Session, "post", refuse)


def scores(tmp_path, name, values, judge=None):
    path = tmp_path / name
    rows = [ScoredRow(i, True, faithfulness=f, answer_correctness=c)
            for i, (f, c) in enumerate(values, 1)]
    write_scores(path, rows, {"judge": judge} if judge else {})
    if judge is None:
        (tmp_path / (name + ".meta.json")).unlink()
    return str(path)


def test_agreement_prints_table_and_both_judges(tmp_path, capsys):
    a = scores(tmp_path, "a.csv", [(0.2, 5), (0.9, 4), (1.0, 2), (0.5, 3)], judge="mistral-x")
    b = scores(tmp_path, "b.csv", [(0.2, 5), (0.9, 3), (1.0, 2), (0.5, 3)], judge="other-y")
    assert main(["agreement", "--a", a, "--b", b]) == 0
    out = capsys.readouterr().out
    assert "| metric | pearson | kappa | threshold | n |" in out
    assert "| faithfulness | 1.000 | 1.000 | 0.8 | 4 |" in out
    assert "| answer_correctness |" in out and "| 4.0 | 4 |" in out
    assert "| context_recall | nan | nan | 0.8 | 0 |" in out
    assert "a: judge mistral-x" in out and "b: judge other-y" in out


def test_agreement_without_sidecar_says_provenance_none(tmp_path, capsys):
    a = scores(tmp_path, "a.csv", [(0.2, 5), (0.9, 4)], judge="mistral-x")
    b = scores(tmp_path, "b.csv", [(0.2, 5), (0.9, 4)])
    assert main(["agreement", "--a", a, "--b", b]) == 0
    assert "b: provenance: none" in capsys.readouterr().out


def test_agreement_missing_file_exits_2(tmp_path, capsys):
    a = scores(tmp_path, "a.csv", [(0.2, 5)], judge="mistral-x")
    assert main(["agreement", "--a", a, "--b", str(tmp_path / "missing.csv")]) == 2
    assert "missing.csv" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["agreement", "--a", a])
    assert e.value.code == 2

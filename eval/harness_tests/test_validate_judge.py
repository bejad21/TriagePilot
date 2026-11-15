import json

import pandas as pd
import pytest

from harness import validate_judge
from harness.backends import BackendError
from harness.judges import JudgeParseError


def make_df(n_adherent, n_other):
    rows = []
    for i in range(n_adherent + n_other):
        adherent = i < n_adherent
        rows.append({
            "id": f"r{i:03d}", "question": f"q{i}",
            "documents": [f"doc a {i}", f"doc b {i}"],
            "response": f"{'ADH' if adherent else 'NON'} answer {i}",
            "adherence_score": adherent, "extra": 1})
    return pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)


class FakeJudge:
    """Marks every claim supported iff the answer text starts with `yes_marker`."""
    name = "fake"

    def __init__(self, yes_marker="ADH", always=None, fail_marker=None, exc=None):
        self.yes_marker = yes_marker
        self.always = always
        self.fail_marker = fail_marker
        self.exc = exc
        self.calls = 0

    def complete(self, system, user, *, temperature=0.0):
        self.calls += 1
        if self.fail_marker and self.fail_marker in user:
            raise self.exc
        supported = self.always if self.always is not None else (
            f"{self.yes_marker} answer" in user)
        return json.dumps({"claims": [{"claim": "c", "supported": supported}]})


def test_sample_is_deterministic_and_balanced():
    df = make_df(10, 10)
    a = validate_judge.load_ragbench_sample(df, 8, seed=1)
    b = validate_judge.load_ragbench_sample(df.iloc[::-1], 8, seed=1)
    assert a == b
    assert len(a) == 8
    assert sum(r["adherent"] for r in a) == 4
    assert set(a[0]) == {"id", "question", "documents", "response", "adherent"}
    assert isinstance(a[0]["documents"], list)
    assert validate_judge.load_ragbench_sample(df, 8, seed=2) != a


def test_sample_fills_from_larger_class_when_one_is_short():
    sample = validate_judge.load_ragbench_sample(make_df(10, 2), 8)
    assert len(sample) == 8
    assert sum(not r["adherent"] for r in sample) == 2


def test_sample_odd_n_and_oversized_n():
    df = make_df(5, 5)
    assert len(validate_judge.load_ragbench_sample(df, 7)) == 7
    assert len(validate_judge.load_ragbench_sample(df, 50)) == 10


def test_sample_missing_columns_raises():
    with pytest.raises(ValueError, match="adherence_score"):
        validate_judge.load_ragbench_sample(make_df(2, 2).drop(columns=["adherence_score"]), 2)


def test_perfect_judge():
    sample = validate_judge.load_ragbench_sample(make_df(6, 6), 12)
    result = validate_judge.evaluate_judge(FakeJudge(), sample)
    assert result["accuracy"] == 1.0
    assert result["kappa"] == 1.0
    assert result["precision_adherent"] == 1.0
    assert result["recall_adherent"] == 1.0
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (6, 0, 6, 0)
    assert result["n"] == 12 and result["errors"] == 0


def test_always_adherent_judge_has_zero_kappa():
    sample = validate_judge.load_ragbench_sample(make_df(6, 6), 12)
    result = validate_judge.evaluate_judge(FakeJudge(always=True), sample)
    assert result["kappa"] == 0.0
    assert result["accuracy"] == 0.5
    assert result["recall_adherent"] == 1.0
    assert result["precision_adherent"] == 0.5
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (6, 6, 0, 0)


def test_threshold_is_inclusive():
    sample = [{"id": "a", "question": "q", "documents": ["d"], "response": "r",
               "adherent": True}]
    judge = FakeJudge(always=True)
    assert validate_judge.evaluate_judge(judge, sample, threshold=1.0)["tp"] == 1


@pytest.mark.parametrize("exc", [JudgeParseError("bad"), BackendError("down")])
def test_errors_are_excluded_not_scored(exc):
    sample = validate_judge.load_ragbench_sample(make_df(4, 4), 8)
    judge = FakeJudge(fail_marker="NON answer", exc=exc)
    result = validate_judge.evaluate_judge(judge, sample)
    assert result["errors"] == 4
    assert result["n"] == 4
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (4, 0, 0, 0)


def test_none_score_counts_as_error():
    class EmptyClaims:
        name = "fake"

        def complete(self, system, user, *, temperature=0.0):
            return json.dumps({"claims": []})

    sample = validate_judge.load_ragbench_sample(make_df(2, 2), 4)
    result = validate_judge.evaluate_judge(EmptyClaims(), sample)
    assert result["errors"] == 4 and result["n"] == 0
    assert result["accuracy"] is None and result["kappa"] is None
    assert result["precision_adherent"] is None and result["recall_adherent"] is None


def test_no_predicted_positives_gives_no_precision():
    sample = validate_judge.load_ragbench_sample(make_df(2, 2), 4)
    result = validate_judge.evaluate_judge(FakeJudge(always=False), sample)
    assert result["precision_adherent"] is None
    assert result["recall_adherent"] == 0.0


def test_cli_requires_key_before_any_fetch(monkeypatch, capsys):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    def no_fetch(subset, split):
        raise AssertionError("network used")

    assert validate_judge.main([], fetch=no_fetch) == 2
    assert "MISTRAL_API_KEY" in capsys.readouterr().err


def test_cli_refuses_existing_out_before_any_fetch(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    out = tmp_path / "r.json"
    out.write_text("keep")

    def no_fetch(subset, split):
        raise AssertionError("network used")

    assert validate_judge.main(["--out", str(out)], fetch=no_fetch) == 2
    assert out.read_text() == "keep"
    assert "exists" in capsys.readouterr().err


def test_cli_runs_with_injected_backend_and_writes_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    cache = tmp_path / "c.jsonl"
    seen = []

    def fetch(subset, split):
        seen.append((subset, split))
        return make_df(6, 6)

    judge = FakeJudge()
    code = validate_judge.main(
        ["--n", "12", "--cache", str(cache), "--out", str(out)],
        fetch=fetch, backend=judge)
    assert code == 0
    assert seen == [("techqa", "test")]
    result = json.loads(out.read_text())
    assert result["accuracy"] == 1.0 and result["n"] == 12
    assert json.loads(capsys.readouterr().out)["kappa"] == 1.0
    assert cache.exists()
    calls = judge.calls
    validate_judge.main(["--n", "12", "--cache", str(cache)], fetch=fetch, backend=judge)
    assert judge.calls == calls


def test_cli_result_carries_run_params_and_no_key(monkeypatch, tmp_path, capsys):
    import datetime
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-fake-secret-123")
    out = tmp_path / "r.json"
    judge = FakeJudge()
    judge.name = "api.example.test/some-model"
    fixed = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    argv = ["--subset", "emanual", "--split", "validation", "--n", "10", "--seed", "7",
            "--threshold", "0.5", "--cache", str(tmp_path / "c.jsonl"), "--out", str(out)]
    code = validate_judge.main(argv, fetch=lambda s, p: make_df(6, 6),
                               backend=judge, clock=lambda: fixed)
    assert code == 0
    text = out.read_text()
    assert "sk-fake-secret-123" not in text
    assert "sk-fake-secret-123" not in capsys.readouterr().out
    assert json.loads(text)["params"] == {
        "subset": "emanual", "split": "validation", "n_requested": 10, "n_sample": 10,
        "seed": 7, "threshold": 0.5, "backend": "api.example.test/some-model",
        "cache": True, "created_utc": "2026-01-02T03:04:05+00:00"}
    assert validate_judge.main(argv, fetch=lambda s, p: make_df(6, 6),
                               backend=judge) == 2


def test_cli_params_record_sample_shortfall_and_no_cache(tmp_path):
    out = tmp_path / "r.json"
    validate_judge.main(["--n", "50", "--out", str(out)],
                        fetch=lambda s, p: make_df(3, 3), backend=FakeJudge())
    params = json.loads(out.read_text())["params"]
    assert (params["n_requested"], params["n_sample"], params["cache"]) == (50, 6, False)
    assert params["created_utc"].endswith("+00:00")


class FakeResponse:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json))
        user = json["messages"][-1]["content"]
        supported = "ADH answer" in user
        return FakeResponse(
            '{"claims": [{"claim": "c", "supported": %s}]}' % str(supported).lower())


def _no_fetch(subset, split):
    raise AssertionError("network used")


def _run_no_network(argv, capsys, session=None):
    session = session or FakeSession()
    code = validate_judge.main(argv, fetch=_no_fetch, session=session)
    assert session.posts == []
    return code, capsys.readouterr().err


def test_base_url_without_key_env_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    code, err = _run_no_network(
        ["--judge-model", "m", "--judge-base-url", "https://api.example.test/v1"], capsys)
    assert code == 2
    assert "--judge-key-env" in err


def test_key_env_without_base_url_is_refused(capsys):
    code, err = _run_no_network(["--judge-model", "m", "--judge-key-env", "X_KEY"], capsys)
    assert code == 2
    assert "go together" in err


def test_missing_env_var_is_named_and_nothing_is_sent(monkeypatch, capsys):
    monkeypatch.delenv("OTHER_JUDGE_KEY", raising=False)
    code, err = _run_no_network(
        ["--judge-model", "m", "--judge-base-url", "https://api.example.test/v1",
         "--judge-key-env", "OTHER_JUDGE_KEY"], capsys)
    assert code == 2
    assert "OTHER_JUDGE_KEY" in err


def test_base_model_required_with_base_url(monkeypatch, capsys):
    monkeypatch.setenv("OTHER_JUDGE_KEY", "k")
    code, err = _run_no_network(
        ["--judge-base-url", "https://api.example.test/v1",
         "--judge-key-env", "OTHER_JUDGE_KEY"], capsys)
    assert code == 2
    assert "--judge-model" in err


@pytest.mark.parametrize("url", ["https://user:pw@api.example.test/v1",
                                 "https://api.example.test/v1?key=abc"])
def test_base_url_with_userinfo_or_query_is_refused(monkeypatch, capsys, url):
    monkeypatch.setenv("OTHER_JUDGE_KEY", "sk-fake-secret-123")
    code, err = _run_no_network(
        ["--judge-model", "m", "--judge-base-url", url,
         "--judge-key-env", "OTHER_JUDGE_KEY"], capsys)
    assert code == 2
    assert "user info" in err
    assert "pw" not in err and "abc" not in err


def test_openai_compat_judge_is_built_and_recorded(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OTHER_JUDGE_KEY", "sk-fake-secret-123")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    session = FakeSession()
    out = tmp_path / "r.json"
    code = validate_judge.main(
        ["--n", "12", "--out", str(out), "--judge-model", "judge-2026-01",
         "--judge-base-url", "https://api.example.test:8443/v1",
         "--judge-key-env", "OTHER_JUDGE_KEY"],
        fetch=lambda s, p: make_df(6, 6), session=session)
    assert code == 0
    assert session.posts
    assert all(url == "https://api.example.test:8443/v1/chat/completions"
               and body["model"] == "judge-2026-01" for url, body in session.posts)
    result = json.loads(out.read_text())
    assert result["accuracy"] == 1.0
    assert result["params"]["backend"] == "api.example.test:8443/judge-2026-01"
    captured = capsys.readouterr()
    for text in (out.read_text(), captured.out, captured.err):
        assert "sk-fake-secret-123" not in text
        assert "OTHER_JUDGE_KEY" not in text


def test_mistral_path_unchanged_with_explicit_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-fake-secret-123")
    session = FakeSession()
    out = tmp_path / "r.json"
    code = validate_judge.main(
        ["--n", "12", "--out", str(out), "--judge-model", "mistral-small-2506"],
        fetch=lambda s, p: make_df(6, 6), session=session)
    assert code == 0
    assert all(url == "https://api.mistral.ai/v1/chat/completions"
               for url, _ in session.posts)
    assert json.loads(out.read_text())["params"]["backend"] == (
        "api.mistral.ai/mistral-small-2506")
    captured = capsys.readouterr()
    assert "not pinned" not in captured.err
    assert "sk-fake-secret-123" not in out.read_text() + captured.out + captured.err


def test_default_alias_prints_unpinned_notice(monkeypatch, capsys):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    session = FakeSession()
    code = validate_judge.main(["--n", "4"], fetch=lambda s, p: make_df(3, 3),
                               session=session)
    assert code == 0
    assert "not pinned" in capsys.readouterr().err
    assert session.posts[0][1]["model"] == "mistral-small-latest"


def test_out_still_refuses_overwrite_with_judge_options(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OTHER_JUDGE_KEY", "k")
    out = tmp_path / "r.json"
    out.write_text("keep")
    code, err = _run_no_network(
        ["--out", str(out), "--judge-model", "m",
         "--judge-base-url", "https://api.example.test/v1",
         "--judge-key-env", "OTHER_JUDGE_KEY"], capsys)
    assert code == 2
    assert out.read_text() == "keep"

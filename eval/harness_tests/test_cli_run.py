import hashlib
import json

import pytest

from harness import HARNESS_VERSION, cli_pipeline
from harness.__main__ import main
from harness.run_artifacts import RunRecord, RunStore

SECRET = "sk-fake-SECRET-value-0123456789"
ANSWERED = {"status": "answered", "answer": "a", "cited_source": "s.md"}


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Answers every POST with the next scripted response (default: answered)."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return self.script.pop(0) if self.script else FakeResponse(ANSWERED)


@pytest.fixture
def session(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(cli_pipeline, "_session", lambda: fake)
    monkeypatch.setenv("MISTRAL_API_KEY", SECRET)
    return fake


def write_data(tmp_path, ids=range(1, 41), name="wix_v2.json"):
    path = tmp_path / name
    rows = [{"id": i, "question": f"q{i}?", "expected_answer": f"a{i}", "gold_sources": ["g"],
             "expected_behavior": "answer"} for i in ids]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def args(data, store, *extra, split="heldout", select="heldout", system="wix"):
    return ["run", "--data", str(data), "--system", system, "--split", split,
            "--select", select, "--store", str(store), *extra]


def manifest(store):
    path = store.parent / (store.name + ".manifest.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_without_yes_prints_plan_and_sends_nothing(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store)) == 0
    out = capsys.readouterr().out
    assert "add --yes to spend model quota" in out
    assert hashlib.sha256(data.read_bytes()).hexdigest() in out
    assert "10 rows" in out and "31" in out and "40" in out
    assert "http://localhost:5678/webhook/triage-wix" in out
    assert session.calls == []
    assert not store.exists() and manifest(store) is None


def test_dry_run_with_yes_sends_nothing(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--dry-run")) == 0
    assert "add --yes" not in capsys.readouterr().out
    assert session.calls == []
    assert not store.exists() and manifest(store) is None


def test_yes_runs_heldout_rows_and_writes_manifest(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    code = main(args(data, store, "--yes", "--config", "model=mistral-small-2506",
                     "--system", "techqa"))
    assert code == 0
    assert [c["json"]["text"] for c in session.calls] == [f"q{i}?" for i in range(31, 41)]
    assert session.calls[0]["url"] == "http://localhost:5678/webhook/triage-techqa"
    records = RunStore(store).load()
    assert {(r.dataset, r.split) for r in records} == {("wix_v2", "heldout")}
    assert records[0].config == {"model": "mistral-small-2506"}
    m = manifest(store)
    assert m["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert (m["system"], m["split"], m["select"]) == ("techqa", "heldout", "heldout")
    assert (m["n_rows"], m["id_min"], m["id_max"]) == (10, 31, 40)
    assert m["webhook_url"] == "http://localhost:5678/webhook/triage-techqa"
    assert m["harness_version"] == HARNESS_VERSION == "1"
    assert m["git_commit"] is None or len(m["git_commit"]) == 40
    assert m["config"] == {"model": "mistral-small-2506"}
    assert m["data_file"] == str(data) and m["created_utc"].endswith("+00:00")


def test_webhook_url_override(tmp_path, session):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "1",
                     "--webhook-url", "http://hook.invalid/x")) == 0
    assert [c["url"] for c in session.calls] == ["http://hook.invalid/x"]


def test_n_limit_keeps_first_rows_after_selection(tmp_path, session):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "3")) == 0
    assert [c["json"]["text"] for c in session.calls] == ["q31?", "q32?", "q33?"]
    assert manifest(store)["n_rows"] == 3


def test_resume_posts_only_missing_rows_and_keeps_the_manifest(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "4")) == 0
    first = (tmp_path / "runs.jsonl.manifest.json").read_bytes()
    session.calls.clear()
    assert main(args(data, store)) == 0
    assert "4 of 10 already done" in capsys.readouterr().out
    assert main(args(data, store, "--yes")) == 0
    assert [c["json"]["text"] for c in session.calls] == [f"q{i}?" for i in range(35, 41)]
    assert (tmp_path / "runs.jsonl.manifest.json").read_bytes() == first
    assert len(RunStore(store).load()) == 10


@pytest.mark.parametrize("change", ["data", "system", "split"])
def test_manifest_mismatch_is_refused_before_any_post(tmp_path, session, capsys, change):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "1", "--split", "real",
                     "--select", "all")) == 0
    session.calls.clear()
    kw = {"split": "real", "select": "all"}
    if change == "data":
        data = write_data(tmp_path, ids=range(1, 42))
    elif change == "system":
        kw["system"] = "techqa"
    else:
        kw["split"] = "other"
        # Records of the old split would also be refused; drop them to test the manifest alone.
        store.unlink()
    assert main(args(data, store, "--yes", **kw)) == 2
    err = capsys.readouterr().err
    assert "manifest" in err and {"data": "data_sha256"}.get(change, change) in err
    assert session.calls == []


def test_store_with_another_split_is_refused_before_manifest_and_post(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    RunStore(store).append(RunRecord(row_id=1, dataset="wix_v2", split="cross", question="q",
                                     expected_behavior="answer", status="answered"))
    assert main(args(data, store, "--yes")) == 2
    assert "one run store per" in capsys.readouterr().err
    assert session.calls == [] and manifest(store) is None


def test_tuned_ids_are_refused_for_a_heldout_split(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", select="all")) == 2
    assert "tuned row ids" in capsys.readouterr().err
    assert session.calls == [] and not store.exists() and manifest(store) is None


def test_repeated_row_ids_are_refused_before_manifest_and_post(tmp_path, session, capsys):
    data, store = write_data(tmp_path, ids=[1, 2, 2]), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", split="real", select="all")) == 2
    assert "repeats row ids" in capsys.readouterr().err
    assert session.calls == [] and manifest(store) is None


def test_bad_rows_are_refused_before_manifest_and_post(tmp_path, session, capsys):
    data, store = write_data(tmp_path, ids=[1, 2]), tmp_path / "runs.jsonl"
    rows = json.loads(data.read_text(encoding="utf-8"))
    rows[1]["expected_behavior"] = "bogus"
    data.write_text(json.dumps(rows), encoding="utf-8")
    assert main(args(data, store, "--yes", split="real", select="all")) == 2
    assert "[2]" in capsys.readouterr().err
    assert session.calls == [] and manifest(store) is None


def test_webhook_url_holding_a_secret_is_refused(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--webhook-url",
                     f"http://hook.invalid/x?token={SECRET}")) == 2
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert session.calls == [] and manifest(store) is None


def test_corrupt_manifest_is_refused(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    (tmp_path / "runs.jsonl.manifest.json").write_text("{", encoding="utf-8")
    assert main(args(data, store, "--yes")) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert session.calls == []


def test_error_records_give_exit_1_and_retry_errors_reruns_them(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    session.script = [FakeResponse(ANSWERED), FakeResponse({"oops": 1}, 500)]
    assert main(args(data, store, "--yes", "--n-limit", "2")) == 1
    assert "1 error record(s)" in capsys.readouterr().out
    session.calls.clear()
    assert main(args(data, store, "--yes", "--n-limit", "2")) == 1
    assert session.calls == []
    assert main(args(data, store, "--yes", "--n-limit", "2", "--retry-errors")) == 0
    assert [c["json"]["text"] for c in session.calls] == ["q32?"]


def test_stop_file_leaves_rows_unrun_and_exit_1(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert main(args(data, store, "--yes")) == 1
    assert "10 selected row(s) not run" in capsys.readouterr().out
    assert session.calls == []


def test_no_key_value_in_output_store_or_manifest(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "2", "--config", "k=6")) == 0
    out = capsys.readouterr()
    for text in (out.out, out.err, store.read_text(encoding="utf-8"),
                 (tmp_path / "runs.jsonl.manifest.json").read_text(encoding="utf-8")):
        assert SECRET not in text


@pytest.mark.parametrize("pair", ["api_key=abc", "auth_token=abc", f"x={SECRET}",
                                  f"x=Bearer {SECRET}", "no-equals-sign"])
def test_config_that_could_hold_a_secret_is_refused(tmp_path, session, capsys, pair):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--config", pair)) == 2
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err
    assert session.calls == [] and manifest(store) is None


def run_log(store):
    path = store.parent / (store.name + ".runs.jsonl")
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_each_invocation_is_logged_and_the_manifest_is_kept(tmp_path, session):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "3")) == 0
    first = (tmp_path / "runs.jsonl.manifest.json").read_bytes()
    assert main(args(data, store, "--yes")) == 0
    assert (tmp_path / "runs.jsonl.manifest.json").read_bytes() == first
    log = run_log(store)
    assert [e["event"] for e in log] == ["started", "finished", "started", "finished"]
    pilot, full = log[0], log[2]
    assert (pilot["n_rows_selected"], pilot["id_min"], pilot["id_max"]) == (3, 31, 33)
    assert (full["n_rows_selected"], full["id_min"], full["id_max"]) == (10, 31, 40)
    assert (pilot["rows_already_done"], full["rows_already_done"]) == (0, 3)
    assert (pilot["n_limit"], full["n_limit"]) == (3, None)
    assert full["select"] == "heldout" and full["retry_errors"] is False
    assert (full["pause"], full["timeout"]) == (0.0, 600.0)
    assert full["harness_version"] == "1" and "git_commit" in full
    assert full["created_utc"].endswith("+00:00")
    assert log[1]["posted"]["answered"] == 3 and log[3]["posted"]["answered"] == 7
    assert log[3]["created_utc"].endswith("+00:00")


def test_changed_webhook_or_config_on_resume_is_noted_and_logged(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "1", "--config", "k=6")) == 0
    capsys.readouterr()
    assert main(args(data, store, "--yes", "--n-limit", "2", "--config", "k=8",
                     "--webhook-url", "http://hook.invalid/other")) == 0
    assert "differs from manifest: webhook_url, config" in capsys.readouterr().out
    started = [e for e in run_log(store) if e["event"] == "started"]
    assert started[1]["webhook_url"] == "http://hook.invalid/other"
    assert started[1]["config"] == {"k": "8"}
    assert manifest(store)["config"] == {"k": "6"}


def test_invocation_is_logged_before_the_first_post(tmp_path, monkeypatch):
    class Exploding:
        def post(self, *a, **kw):
            raise RuntimeError("crash")

    monkeypatch.setattr(cli_pipeline, "_session", Exploding)
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    with pytest.raises(RuntimeError):
        main(args(data, store, "--yes"))
    log = run_log(store)
    assert [e["event"] for e in log] == ["started"] and log[0]["n_rows_selected"] == 10


def test_without_yes_or_with_dry_run_no_log_is_written(tmp_path, session):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store)) == 0
    assert main(args(data, store, "--yes", "--dry-run")) == 0
    assert run_log(store) is None and manifest(store) is None


def test_no_key_value_in_run_log(tmp_path, session):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "2", "--config", "k=6")) == 0
    text = (tmp_path / "runs.jsonl.runs.jsonl").read_text(encoding="utf-8")
    assert SECRET not in text and '"k": "6"' in text


@pytest.mark.parametrize("split", ["Heldout", " HELDOUT "])
def test_split_is_normalised_before_the_tuned_check(tmp_path, session, capsys, split):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", split=split, select="all")) == 2
    assert "tuned row ids" in capsys.readouterr().err
    assert session.calls == [] and manifest(store) is None


def test_normalised_split_is_stored_everywhere(tmp_path, session):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--n-limit", "1", split=" HeldOut ")) == 0
    assert {r.split for r in RunStore(store).load()} == {"heldout"}
    assert manifest(store)["split"] == "heldout"
    assert run_log(store)[0]["split"] == "heldout"
    assert main(args(data, store, "--yes", "--n-limit", "2", split="heldout")) == 0


@pytest.mark.parametrize("url, hidden", [
    ("http://user:pa55word@hook.invalid/x", "pa55word"),
    ("http://tok3n@hook.invalid/x", "tok3n"),
    ("http://hook.invalid/x?sig=abc123", "abc123"),
    ("http://hook.invalid/x?", None),
])
def test_webhook_url_with_userinfo_or_query_is_refused(tmp_path, session, capsys, url, hidden):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    assert main(args(data, store, "--yes", "--webhook-url", url)) == 2
    out = capsys.readouterr()
    assert "webhook URL" in out.err
    if hidden:
        assert hidden not in out.out + out.err
    assert session.calls == [] and manifest(store) is None and run_log(store) is None


def test_usage_errors_exit_2(tmp_path, session, capsys):
    data, store = write_data(tmp_path), tmp_path / "runs.jsonl"
    with pytest.raises(SystemExit) as e:
        main(["run", "--data", str(data), "--split", "heldout", "--select", "all",
              "--store", str(store)])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(args(data, store, "--n-limit", "0"))
    assert e.value.code == 2
    assert main(args(tmp_path / "missing.json", store, "--yes")) == 2
    assert session.calls == []

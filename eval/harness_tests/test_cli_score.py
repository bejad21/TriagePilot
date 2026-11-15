import argparse
import json

import pytest
import requests

from harness import cli_pipeline
from harness.__main__ import main
from harness.backends import BackendError, CachedChat, MistralChat, MistralEmbed, OpenAICompatChat
from harness.run_artifacts import RunRecord, RunStore
from harness.score_runs import read_meta, read_scores

SECRET = "sk-fake-SECRET-value-0123456789"


class FakeChat:
    name = "fake-judge-2026-01-01"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def complete(self, system, user, *, temperature=0.0):
        self.calls += 1
        if self.fail:
            raise BackendError("quota")
        if "Chunk 1" in user:
            return json.dumps({"relevance": [True, False]})
        if '"attributed"' in user:
            return json.dumps({"claims": [{"claim": "a", "attributed": True}]})
        if '"supported"' in user:
            return json.dumps({"claims": [{"claim": "a", "supported": True}]})
        if '"score"' in user:
            return json.dumps({"score": 4})
        return json.dumps({"questions": ["q1"]})


class FakeEmbedder:
    name = "fake-embed"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Built:
    """Stands in for cli_pipeline._backends and keeps every pair it hands out."""

    def __init__(self):
        self.pairs = []
        self.fail = False

    def __call__(self, args):
        self.pairs.append((FakeChat(fail=self.fail), FakeEmbedder()))
        return self.pairs[-1]


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", SECRET)
    fake = Built()
    monkeypatch.setattr(cli_pipeline, "_backends", fake)
    return fake


def rec(row_id, status, behavior="answer", collected=True, **kw):
    base = dict(row_id=row_id, dataset="wix_v2", split="heldout", question=f"question {row_id}",
                expected_behavior=behavior, expected_answer="ref", gold_sources=["g"],
                status=status, answer="ans" if status == "answered" else None,
                retrieved=[{"source": "x", "text": "tx"}, {"source": "g", "text": "tg"}]
                if collected else [], context_collected=collected,
                started_at="2026-08-26T07:21:00+00:00")
    base.update(kw)
    return RunRecord(**base)


CLEAN = [rec(1, "answered"), rec(2, "escalated"), rec(3, "answered", behavior="escalate",
                                                   collected=False), rec(4, "rejected")]


def setup(tmp_path, records=CLEAN, *extra):
    store = RunStore(tmp_path / "runs.jsonl")
    for r in records:
        store.append(r)
    out = tmp_path / "scores.csv"
    return out, ["score", "--store", str(store.path), "--out", str(out),
                 "--judge-model", "judge-x", *extra]


def test_score_writes_scores_and_sidecar_and_prints_tables(tmp_path, built, capsys):
    out, argv = setup(tmp_path)
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert [r.row_id for r in read_scores(out)] == [1, 2, 4]
    meta = read_meta(out)
    assert meta["judge"] == "fake-judge-2026-01-01" and meta["embedder"] == "fake-embed"
    assert meta["ctx_k"] == 6 and meta["runs"] == argv[2]
    assert "judge errors 0" in printed and "run errors 0" in printed
    assert "provenance: judge fake-judge-2026-01-01" in printed
    assert "## Scores, split: heldout" in printed and "## Abstention behavior" in printed
    assert built.pairs[0][0].calls > 0


def test_ctx_k_is_passed_and_recorded(tmp_path, built):
    out, argv = setup(tmp_path)
    assert main(argv + ["--ctx-k", "1"]) == 0
    assert read_meta(out)["ctx_k"] == 1
    assert read_scores(out)[0].hit_at_k == 0.0


def test_judge_failures_do_not_abort_and_are_counted(tmp_path, built, capsys):
    out, argv = setup(tmp_path)
    built.fail = True
    assert main(argv) == 0
    assert "judge errors 0" not in capsys.readouterr().out
    assert len(read_meta(out)["judge_errors"]) > 0


def test_no_key_value_in_output_or_files(tmp_path, built, capsys):
    out, argv = setup(tmp_path)
    assert main(argv) == 0
    printed = capsys.readouterr()
    for text in (printed.out, printed.err, out.read_text(encoding="utf-8"),
                 (tmp_path / "scores.csv.meta.json").read_text(encoding="utf-8")):
        assert SECRET not in text


def test_judge_model_is_required(tmp_path, built):
    _, argv = setup(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(argv[:-2])
    assert e.value.code == 2


@pytest.mark.parametrize("name", ["scores.csv", "scores.csv.meta.json"])
def test_existing_output_is_refused_before_any_backend(tmp_path, built, capsys, name):
    out, argv = setup(tmp_path)
    (tmp_path / name).write_text("keep", encoding="utf-8")
    assert main(argv) == 1
    assert "already exists" in capsys.readouterr().err
    assert built.pairs == [] and (tmp_path / name).read_text(encoding="utf-8") == "keep"


def test_missing_output_directory_is_refused(tmp_path, built, capsys):
    _, argv = setup(tmp_path)
    argv[argv.index("--out") + 1] = str(tmp_path / "nowhere" / "scores.csv")
    assert main(argv) == 2
    assert "nowhere" in capsys.readouterr().err and built.pairs == []


def test_missing_mistral_key_names_the_variable(tmp_path, built, capsys, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY")
    _, argv = setup(tmp_path)
    assert main(argv) == 1
    assert "MISTRAL_API_KEY" in capsys.readouterr().err and built.pairs == []


def test_custom_judge_needs_key_env_and_names_it_when_unset(tmp_path, built, capsys,
                                                            monkeypatch):
    monkeypatch.delenv("OTHER_JUDGE_KEY", raising=False)
    _, argv = setup(tmp_path)
    assert main(argv + ["--judge-base-url", "https://judge.invalid/v1"]) == 2
    assert "--judge-key-env" in capsys.readouterr().err
    assert main(argv + ["--judge-key-env", "OTHER_JUDGE_KEY"]) == 2
    capsys.readouterr()
    assert main(argv + ["--judge-base-url", "https://judge.invalid/v1",
                        "--judge-key-env", "OTHER_JUDGE_KEY"]) == 1
    err = capsys.readouterr().err
    assert "OTHER_JUDGE_KEY" in err and SECRET not in err and built.pairs == []


def test_uncollected_answer_rows_are_refused_unless_allowed(tmp_path, built, capsys):
    records = CLEAN + [rec(5, "answered", collected=False), rec(6, "escalated", collected=False)]
    out, argv = setup(tmp_path, records)
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "2 answered or escalated" in err and "collect" in err
    assert built.pairs == [] and not out.exists()
    assert main(argv + ["--allow-uncollected"]) == 0
    by_id = {r.row_id: r for r in read_scores(out)}
    assert by_id[5].faithfulness is None and by_id[1].faithfulness is not None


def test_retrieved_without_flag_counts_as_collected(tmp_path, built):
    out, argv = setup(tmp_path, [rec(1, "answered", context_collected=False)])
    assert main(argv) == 0


def test_error_records_are_refused_unless_allowed(tmp_path, built, capsys):
    out, argv = setup(tmp_path, CLEAN + [rec(7, "error", error="ReadTimeout")])
    assert main(argv) == 1
    assert "--allow-errors" in capsys.readouterr().err and built.pairs == []
    assert main(argv + ["--allow-errors"]) == 0
    printed = capsys.readouterr().out
    assert "run errors 1" in printed and "run errors excluded: 1" in printed


def test_missing_or_empty_store_exits_2(tmp_path, built, capsys):
    _, argv = setup(tmp_path, [])
    assert main(argv) == 2
    argv[argv.index("--store") + 1] = str(tmp_path / "missing.jsonl")
    assert main(argv) == 2
    assert "missing.jsonl" in capsys.readouterr().err and built.pairs == []


def test_refusals_make_no_request_with_the_real_backends(tmp_path, monkeypatch, capsys):
    def no_network(*a, **kw):
        raise AssertionError("network call")

    monkeypatch.setattr(requests.Session, "post", no_network)
    monkeypatch.setenv("MISTRAL_API_KEY", SECRET)
    _, argv = setup(tmp_path, CLEAN + [rec(5, "answered", collected=False)])
    assert main(argv) == 1
    (tmp_path / "e").mkdir()
    _, argv = setup(tmp_path / "e", CLEAN + [rec(7, "error")])
    assert main(argv) == 1
    monkeypatch.delenv("MISTRAL_API_KEY")
    assert main(argv + ["--allow-errors"]) == 1


def ns(**kw):
    base = dict(judge_model="judge-x", judge_base_url=None, judge_key_env=None,
                judge_extra_body=None, embed_model="mistral-embed", cache=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_backends_are_built_from_the_flags(tmp_path):
    chat, embedder = cli_pipeline._backends(ns())
    assert isinstance(chat, MistralChat) and chat.model == "judge-x"
    assert isinstance(embedder, MistralEmbed) and embedder.model == "mistral-embed"
    chat, _ = cli_pipeline._backends(ns(judge_base_url="https://judge.invalid/v1",
                                        judge_key_env="OTHER_JUDGE_KEY"))
    assert type(chat) is OpenAICompatChat and chat.api_key_env == "OTHER_JUDGE_KEY"
    assert chat.name == "judge.invalid/judge-x"
    chat, _ = cli_pipeline._backends(ns(judge_base_url="https://judge.invalid/v1",
                                        judge_key_env="OTHER_JUDGE_KEY",
                                        judge_extra_body='{"think": false}'))
    assert chat.extra_body == {"think": False}
    chat, _ = cli_pipeline._backends(ns(cache=str(tmp_path / "cache.jsonl")))
    assert isinstance(chat, CachedChat) and isinstance(chat.inner, MistralChat)


def test_error_record_is_refused_by_score_and_report_unless_allowed(tmp_path, built, capsys):
    converted = rec(8, "error", collected=False, question="blocked?",
                    error="n8n node error: Guardrails Chat Model: Status 429 Rate limit exceeded")
    out, argv = setup(tmp_path, CLEAN + [converted])
    assert main(argv) == 1
    assert "1 run error record(s)" in capsys.readouterr().err and built.pairs == []
    assert main(argv + ["--allow-errors"]) == 0
    assert 8 not in [r.row_id for r in read_scores(out)]
    assert "run errors 1" in capsys.readouterr().out
    report_argv = ["report", "--runs", argv[2], "--scores", str(out)]
    assert main(report_argv) == 1
    assert "1 run error record(s)" in capsys.readouterr().err
    assert main(report_argv + ["--allow-errors"]) == 0
    assert "run errors excluded: 1" in capsys.readouterr().out

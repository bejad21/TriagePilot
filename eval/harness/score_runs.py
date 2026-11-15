import csv
import dataclasses
import io
import json
import os
import pathlib
from datetime import datetime, timezone
from typing import Callable, Sequence

from harness import judges, metrics
from harness.backends import BackendError
from harness.judges import ChatBackend, EmbeddingBackend, JudgeParseError
from harness.metrics import ScoredRow
from harness.run_artifacts import RunRecord

_METRIC_FIELDS = [f.name for f in dataclasses.fields(ScoredRow)
                  if f.name not in ("row_id", "answered")]


def _judge(row_id: int, name: str, call: Callable[[], float | None],
           errors: list[tuple[int, str, str]]) -> float | None:
    try:
        return call()
    except (JudgeParseError, BackendError) as e:
        errors.append((row_id, name, str(e)))
        return None


def _score_one(rec: RunRecord, chat: ChatBackend, embedder: EmbeddingBackend,
               ctx_k: int, errors: list[tuple[int, str, str]]) -> ScoredRow:
    """Score one record; context metrics stay None when its context was never collected.

    Context counts as available when `context_collected` is set or `retrieved` is
    non-empty. Without it faithfulness, context precision and recall, hit_at_k and
    reciprocal_rank are left None (excluded, not scored 0). The judges and the
    retrieval metrics see the same chunks: the first `ctx_k` of `retrieved`, ranked by
    reranker score, so hit_at_k is Hit@ctx_k.
    """
    answered = rec.status == "answered"
    if not answered and rec.status != "escalated":
        return ScoredRow(rec.row_id, False)
    rid = rec.row_id
    has_context = rec.context_collected or bool(rec.retrieved)
    contexts = [c["text"] for c in rec.retrieved[:ctx_k]]
    values: dict[str, float | None] = {}
    if answered and rec.answer:
        if has_context:
            values["faithfulness"] = _judge(
                rid, "faithfulness",
                lambda: judges.score_faithfulness(chat, rec.answer, contexts), errors)
        values["answer_relevancy"] = _judge(
            rid, "answer_relevancy",
            lambda: judges.score_answer_relevancy(chat, embedder, rec.question, rec.answer),
            errors)
        if rec.expected_answer:
            values["answer_correctness"] = _judge(
                rid, "answer_correctness",
                lambda: judges.score_answer_correctness(
                    chat, rec.question, rec.expected_answer, rec.answer), errors)
    if rec.expected_answer and rec.retrieved:
        values["context_precision"] = _judge(
            rid, "context_precision",
            lambda: judges.score_context_precision(
                chat, rec.question, rec.expected_answer, contexts), errors)
        values["context_recall"] = _judge(
            rid, "context_recall",
            lambda: judges.score_context_recall(chat, rec.expected_answer, contexts), errors)
    if has_context:
        sources = [c["source"] for c in rec.retrieved[:ctx_k]]
        gold = set(rec.gold_sources)
        values["hit_at_k"] = metrics.hit_at_k(sources, gold)
        values["reciprocal_rank"] = metrics.reciprocal_rank(sources, gold)
    return ScoredRow(rid, answered, **values)


def score_records_with_errors(
    records: Sequence[RunRecord], chat: ChatBackend, embedder: EmbeddingBackend,
    *, ctx_k: int = 6,
) -> tuple[list[ScoredRow], list[tuple[int, str, str]]]:
    """Score answerable rows; return the rows and (row id, metric, message) errors.

    A failed judge leaves its metric None. Rejected rows count as not answered and
    trigger no judge calls. Runner error rows are left out of the scored rows and
    listed as errors with metric "run" and the record's error text.
    """
    errors: list[tuple[int, str, str]] = []
    rows = []
    for rec in records:
        if rec.expected_behavior != "answer":
            continue
        if rec.status == "error":
            errors.append((rec.row_id, "run", rec.error or ""))
            continue
        rows.append(_score_one(rec, chat, embedder, ctx_k, errors))
    return rows, errors


def score_records(records: Sequence[RunRecord], chat: ChatBackend,
                  embedder: EmbeddingBackend, *, ctx_k: int = 6) -> list[ScoredRow]:
    """Score answerable rows, discarding the judge error list."""
    return score_records_with_errors(records, chat, embedder, ctx_k=ctx_k)[0]


def _cell(value: float | None) -> str:
    return "" if value is None else repr(value)


def meta_path(path: str | os.PathLike) -> pathlib.Path:
    """The provenance sidecar of a scores CSV: `<path>.meta.json`."""
    return pathlib.Path(str(path) + ".meta.json")


def scores_meta(chat: ChatBackend, embedder: EmbeddingBackend, *, ctx_k: int,
                runs_path: str | os.PathLike,
                errors: Sequence[tuple[int, str, str]]) -> dict:
    """Provenance for write_scores: backends, ctx_k, runs file, time and judge errors.

    Run errors (metric "run") are not judge errors and are left out.
    """
    return {
        "judge": chat.name,
        "embedder": embedder.name,
        "ctx_k": ctx_k,
        "runs": str(runs_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "judge_errors": [{"row_id": rid, "metric": metric, "message": message}
                         for rid, metric, message in errors if metric != "run"],
    }


def _write_new(path: pathlib.Path, text: str, created: list[pathlib.Path]) -> None:
    with open(path, "x", newline="", encoding="utf-8") as f:
        created.append(path)
        f.write(text)


def write_scores(path: str | os.PathLike, rows: Sequence[ScoredRow], meta: dict) -> None:
    """Write scored rows as CSV plus the `meta` sidecar; None metrics become blank cells.

    Both files are created exclusively: if either exists nothing is written, and a
    failure part way removes only the files this call created.
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["row_id", "answered", *_METRIC_FIELDS])
    for r in rows:
        writer.writerow([r.row_id, r.answered,
                         *(_cell(getattr(r, n)) for n in _METRIC_FIELDS)])
    outputs = [(pathlib.Path(path), buf.getvalue()),
               (meta_path(path), json.dumps(meta, indent=2, ensure_ascii=False) + "\n")]
    for target, _ in outputs:
        if target.exists():
            raise FileExistsError(f"{target} already exists; refusing to overwrite")
    created: list[pathlib.Path] = []
    try:
        for target, text in outputs:
            _write_new(target, text, created)
    except BaseException:
        for target in created:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def read_meta(path: str | os.PathLike) -> dict | None:
    """The sidecar written beside a scores CSV, or None if there is none."""
    sidecar = meta_path(path)
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def read_scores(path: str | os.PathLike) -> list[ScoredRow]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            ScoredRow(int(raw["row_id"]), raw["answered"] == "True",
                      **{n: float(raw[n]) if raw[n] else None for n in _METRIC_FIELDS})
            for raw in csv.DictReader(f)
        ]

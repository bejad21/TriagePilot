"""Resumable runner that sends data-set rows to the live triage webhook.

Use one run store per (dataset, split). Row ids repeat across data files (the
cross-domain sets reuse the held-out ids), and the store is keyed by row id, so
`run_rows` refuses a store that already holds another dataset or split.
"""
import json
import os
import pathlib
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Sequence

import requests

from .run_artifacts import STATUSES, RunRecord, RunStore, validate_record

MAX_ERROR_CHARS = 300
REFUSED_RETRY_DELAY_S = 1.0


def load_rows_json(path: str | pathlib.Path) -> list[dict]:
    """Load a JSON list of row objects; each needs an `id` and a `question`."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of rows")
    for i, row in enumerate(data):
        if not isinstance(row, dict) or "id" not in row or "question" not in row:
            raise ValueError(f"{path}: item {i} must be an object with 'id' and 'question'")
    return data


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(text: str) -> str:
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS] + "..."


def new_session() -> requests.Session:
    """A session that sends the webhook secret (X-TriagePilot-Key) when TRIAGEPILOT_WEBHOOK_KEY is set."""
    session = requests.Session()
    key = os.environ.get("TRIAGEPILOT_WEBHOOK_KEY")
    if key:
        session.headers["X-TriagePilot-Key"] = key
    return session


def _was_refused(exc: BaseException | None, depth: int = 0) -> bool:
    """True if a connection-refused error sits anywhere in the exception chain."""
    if exc is None or depth > 8:
        return False
    if isinstance(exc, ConnectionRefusedError):
        return True
    inner = [exc.__cause__, exc.__context__, getattr(exc, "reason", None), *exc.args]
    return any(isinstance(e, BaseException) and _was_refused(e, depth + 1) for e in inner)


def _post(session, url: str, question: str, timeout: float, sleep: Callable[[float], None]):
    # Refused means nothing reached the server, so one retry costs no model quota.
    try:
        return session.post(url, json={"text": question}, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        if isinstance(exc, requests.exceptions.Timeout) or not _was_refused(exc):
            raise
        sleep(REFUSED_RETRY_DELAY_S)
        return session.post(url, json={"text": question}, timeout=timeout)


def _fill_from_body(record: RunRecord, body) -> None:
    status = body.get("status") if isinstance(body, dict) else None
    if status == "answered" and isinstance(body.get("answer"), str):
        record.status = "answered"
        record.answer = body["answer"]
        record.cited_source = body.get("cited_source")
    elif status == "escalated" and isinstance(body.get("message"), str):
        record.status = "escalated"
        record.reason = body["message"]
    elif status == "rejected" and isinstance(body.get("reason"), str):
        record.status = "rejected"
        record.reason = body["reason"]
    else:
        record.status = "error"
        record.error = _short(f"unexpected response body: {json.dumps(body)}")


def _call(record: RunRecord, session, url: str, timeout: float, sleep) -> None:
    try:
        response = _post(session, url, record.question, timeout, sleep)
    except requests.exceptions.RequestException as exc:
        record.status = "error"
        record.error = _short(f"{type(exc).__name__}: {exc}")
        return
    if response.status_code >= 400:
        record.status = "error"
        record.error = _short(f"HTTP {response.status_code}: {response.text}")
        return
    try:
        body = response.json()
    except ValueError:
        record.status = "error"
        record.error = _short(f"malformed JSON: {response.text}")
        return
    _fill_from_body(record, body)


def run_rows(rows: Sequence[dict], webhook_url: str, store: RunStore, *, dataset: str,
             split: str, config: dict, timeout: float = 600.0, pause_s: float = 0.0,
             session=None, sleep: Callable[[float], None] = time.sleep,
             now: Callable[[], str] = _utc_now, clock: Callable[[], float] = time.monotonic,
             retry_errors: bool = False) -> dict:
    """POST each unfinished row once and append one record per row to the store.

    A row counts as finished once the store holds a record for it; with
    `retry_errors` a row whose latest record is an error runs again. A file named
    STOP next to the store halts the loop before the next row. Returns the number
    of rows recorded in this call by status.

    Raises ValueError before any POST if `rows` repeats an id or the store already
    holds records of another dataset or split.
    """
    dupes = sorted(i for i, n in Counter(row["id"] for row in rows).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate row ids: {dupes}")
    other = {(r.dataset, r.split) for r in store} - {(dataset, split)}
    if other:
        raise ValueError(f"{store.path} holds records for {sorted(other)}, not "
                         f"{(dataset, split)}; use one run store per (dataset, split)")
    session = session or new_session()
    stop_file = store.path.with_name("STOP")
    if retry_errors:
        done = {r.row_id for r in store.load() if r.status != "error"}
    else:
        done = store.done_ids()
    counts = {s: 0 for s in STATUSES}
    first = True
    for row in rows:
        if row["id"] in done:
            continue
        if stop_file.exists():
            break
        record = RunRecord(
            row_id=row["id"], dataset=dataset, split=split, question=row["question"],
            expected_behavior=row.get("expected_behavior", "answer"),
            expected_answer=row.get("expected_answer"),
            gold_sources=list(row.get("gold_sources", [])), config=config)
        record.status = "error"
        validate_record(record)  # fail before spending a model call on a bad row
        if pause_s and not first:
            sleep(pause_s)
        first = False
        record.started_at = now()
        t0 = clock()
        _call(record, session, webhook_url, timeout, sleep)
        record.duration_s = clock() - t0
        store.append(record)
        counts[record.status] += 1
    return counts

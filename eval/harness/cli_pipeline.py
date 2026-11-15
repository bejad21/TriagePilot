"""Pipeline subcommands of `python -m harness`: run, score and agreement.

Every refusal happens before the first network call, so a bad invocation costs
no model quota.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

from harness import HARNESS_VERSION, report
from harness.abstention_sets import assert_not_tuned, heldout_rows
from harness.backends import CachedChat, MistralEmbed
from harness.judge_select import (build_judge, judge_args_error, parse_extra_body,
                                  unrecordable_url_error)
from harness.run_artifacts import BEHAVIORS, RunStore
from harness.runner import load_rows_json, new_session, run_rows
from harness.score_runs import (
    meta_path, read_meta, read_scores, score_records_with_errors, scores_meta, write_scores)

REPO = pathlib.Path(__file__).resolve().parents[2]
WEBHOOKS = {"wix": "http://localhost:5678/webhook/triage-wix",
            "techqa": "http://localhost:5678/webhook/triage-techqa"}
MANIFEST_KEYS = ("data_sha256", "system", "split")
RESUME_NOTICE_KEYS = ("webhook_url", "select", "config")
_SECRET_NAME = re.compile(r"key|token|secret|password|auth", re.IGNORECASE)


def _session():
    return new_session()


def _fail(message: str, code: int) -> int:
    print(message, file=sys.stderr)
    return code


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _split_name(text: str) -> str:
    """Stripped, lower-case split, so `Heldout` cannot skip the held-out checks."""
    name = text.strip().lower()
    if not name:
        raise argparse.ArgumentTypeError("must not be empty")
    return name


def manifest_path(store: str | os.PathLike) -> pathlib.Path:
    """The manifest beside a run store: `<store>.manifest.json`."""
    return pathlib.Path(str(store) + ".manifest.json")


def run_log_path(store: str | os.PathLike) -> pathlib.Path:
    """The append-only log of `run --yes` invocations: `<store>.runs.jsonl`."""
    return pathlib.Path(str(store) + ".runs.jsonl")


def _append_json(path: pathlib.Path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _git_commit() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else None


def _secret_values() -> list[str]:
    """Values of environment variables whose names suggest a key or token."""
    return [v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 8]


def _parse_config(pairs: list[str]) -> dict:
    """KEY=VALUE pairs; ValueError for a malformed pair or one that could hold a secret."""
    secrets = _secret_values()
    config = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        # The pair is not echoed: a malformed one may be a pasted secret.
        if not sep or not key:
            raise ValueError("--config entries must be KEY=VALUE")
        if _SECRET_NAME.search(key) or any(s in value for s in secrets):
            raise ValueError(f"--config {key}: keys and secrets are never recorded")
        config[key] = value
    return config


def add_run_parser(sub) -> None:
    p = sub.add_parser("run", help="send data-set rows to the triage webhook")
    p.add_argument("--data", required=True)
    p.add_argument("--system", required=True, choices=sorted(WEBHOOKS))
    p.add_argument("--split", required=True, type=_split_name)
    p.add_argument("--select", required=True, choices=["heldout", "all"])
    p.add_argument("--store", required=True)
    p.add_argument("--webhook-url")
    p.add_argument("--n-limit", type=_positive_int)
    p.add_argument("--pause", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--config", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--yes", action="store_true", help="spend model quota")
    p.add_argument("--dry-run", action="store_true")


def run(args) -> int:
    """Plan, or with --yes execute, a resumable run of the selected rows."""
    data = pathlib.Path(args.data)
    store = RunStore(args.store)
    try:
        rows = load_rows_json(data)
        config = _parse_config(args.config)
        if args.select == "heldout":
            rows = heldout_rows(rows)
        if args.split == "heldout":
            assert_not_tuned(rows)
        records = store.load()
    except (OSError, ValueError, KeyError) as e:
        return _fail(f"run: {e}", 2)
    rows = rows[:args.n_limit] if args.n_limit else rows
    if not rows:
        return _fail("run: no rows selected", 2)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        return _fail(f"run: {data} repeats row ids", 2)
    bad = [r["id"] for r in rows if not r["question"]
           or r.get("expected_behavior", "answer") not in BEHAVIORS]
    if bad:
        return _fail(f"run: rows {bad} have an empty question or an unknown behavior", 2)
    url = args.webhook_url or WEBHOOKS[args.system]
    # The URL is recorded, so credentials in it are refused without being echoed.
    problem = unrecordable_url_error(url, "webhook URL")
    if problem:
        return _fail(f"run: {problem}", 2)
    if any(s in url for s in _secret_values()):
        return _fail("run: the webhook URL holds a secret; it would be recorded", 2)
    dataset = data.stem
    other = {(r.dataset, r.split) for r in records} - {(dataset, args.split)}
    if other:
        return _fail(f"run: {store.path} holds records for {sorted(other)}, not "
                     f"{(dataset, args.split)}; use one run store per (dataset, split)", 2)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "data_file": str(data),
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "system": args.system, "split": args.split, "select": args.select,
        "n_rows": len(rows), "id_min": min(ids), "id_max": max(ids), "webhook_url": url,
        "git_commit": _git_commit(), "harness_version": HARNESS_VERSION, "config": config,
    }
    mpath = manifest_path(store.path)
    changed = []
    if mpath.exists():
        try:
            old = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError as e:
            return _fail(f"run: {mpath} is not valid JSON: {e}", 2)
        diff = [k for k in MANIFEST_KEYS if old.get(k) != manifest[k]]
        if diff:
            return _fail(f"run: {mpath} records a different {', '.join(diff)}; "
                         "use a new store for a new data file, system or split", 2)
        changed = [k for k in RESUME_NOTICE_KEYS if old.get(k) != manifest[k]]
    latest = {r.row_id: r.status for r in records}
    done = sum(latest.get(i) not in (None, "error") for i in ids)
    errors = sum(latest.get(i) == "error" for i in ids)
    print(f"data: {data} (sha256 {manifest['data_sha256']})\n"
          f"rows: {len(rows)} rows selected, ids {min(ids)}..{max(ids)}, "
          f"select {args.select}, split {args.split}\n"
          f"store: {store.path} ({done} of {len(rows)} already done, {errors} error)\n"
          f"webhook: {url}\npause {args.pause} s, timeout {args.timeout} s")
    if args.dry_run or not args.yes:
        if not args.yes:
            print("add --yes to spend model quota")
        return 0
    if changed:
        print(f"differs from manifest: {', '.join(changed)} (the run log records the new values)")
    if not mpath.exists():
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "x", encoding="utf-8") as f:
            f.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log = run_log_path(store.path)
    # Logged before the first POST so a crashed invocation still leaves its line.
    _append_json(log, {
        "event": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "select": args.select, "n_rows_selected": len(rows), "id_min": min(ids),
        "id_max": max(ids), "rows_already_done": done, "webhook_url": url, "config": config,
        "git_commit": manifest["git_commit"], "harness_version": HARNESS_VERSION,
        "n_limit": args.n_limit, "retry_errors": args.retry_errors, "pause": args.pause,
        "timeout": args.timeout})
    try:
        counts = run_rows(rows, url, store, dataset=dataset, split=args.split, config=config,
                          timeout=args.timeout, pause_s=args.pause, session=_session(),
                          retry_errors=args.retry_errors)
    except ValueError as e:
        return _fail(f"run: {e}", 2)
    _append_json(log, {"event": "finished",
                       "created_utc": datetime.now(timezone.utc).isoformat(), "posted": counts})
    print("recorded this call: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    latest = {r.row_id: r.status for r in store.load()}
    n_errors = sum(s == "error" for s in latest.values())
    missing = sum(i not in latest for i in ids)
    if n_errors:
        print(f"{n_errors} error record(s) in {store.path}; --retry-errors re-runs them")
    if missing:
        print(f"{missing} selected row(s) not run (a STOP file halts the run); "
              "rerun to resume")
    return 1 if n_errors or missing else 0


def add_score_parser(sub) -> None:
    p = sub.add_parser("score", help="score a run store with the LLM judges")
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge-model", required=True, help="a pinned, dated model id")
    p.add_argument("--judge-base-url", help="OpenAI-compatible endpoint; default Mistral")
    p.add_argument("--judge-key-env", help="variable holding the key for --judge-base-url")
    p.add_argument("--judge-extra-body", help="JSON object merged into every judge request")
    p.add_argument("--embed-model", default="mistral-embed")
    p.add_argument("--ctx-k", type=_positive_int, default=6)
    p.add_argument("--cache", help="judge reply cache (JSONL)")
    p.add_argument("--allow-uncollected", action="store_true",
                   help="score answer rows that have no retrieved context")
    p.add_argument("--allow-errors", action="store_true",
                   help="score even if the store holds error records")


def _needed_env(args) -> list[str]:
    names = ["MISTRAL_API_KEY"]  # the embedder always uses Mistral
    if args.judge_base_url and args.judge_key_env not in names:
        names.append(args.judge_key_env)
    return names


def _backends(args):
    """Judge chat (cached with --cache) and embedder; constructing them sends nothing."""
    chat = build_judge(args.judge_model, args.judge_base_url, args.judge_key_env,
                       extra_body=parse_extra_body(args.judge_extra_body))
    if args.cache:
        chat = CachedChat(chat, args.cache)
    return chat, MistralEmbed(args.embed_model)


def score(args) -> int:
    """Score the store into a new CSV and sidecar after every check passes."""
    problem = judge_args_error(args.judge_base_url, args.judge_key_env)
    if problem:
        return _fail(f"score: {problem}", 2)
    out = pathlib.Path(args.out)
    for target in (out, meta_path(out)):
        if target.exists():
            return _fail(f"score: {target} already exists; choose a new --out", 1)
    if not out.parent.is_dir():
        return _fail(f"score: directory {out.parent} does not exist", 2)
    unset = [name for name in _needed_env(args) if not os.environ.get(name)]
    if unset:
        return _fail(f"score: environment variable not set: {', '.join(unset)}", 1)
    store = RunStore(args.store)
    if not store.path.exists():
        return _fail(f"score: {store.path} does not exist", 2)
    try:
        records = store.load()
    except ValueError as e:
        return _fail(f"score: {e}", 2)
    if not records:
        return _fail(f"score: {store.path} holds no records", 2)
    uncollected = sum(r.status in ("answered", "escalated") and r.expected_behavior == "answer"
                      and not r.context_collected and not r.retrieved for r in records)
    if uncollected and not args.allow_uncollected:
        return _fail(f"score: {uncollected} answered or escalated answer row(s) have no "
                     "retrieved context; pass --allow-uncollected to score them without context metrics", 1)
    n_errors = sum(r.status == "error" for r in records)
    if n_errors and not args.allow_errors:
        return _fail(f"score: {n_errors} run error record(s); rerun them with "
                     "`run --retry-errors` or pass --allow-errors", 1)
    try:
        chat, embedder = _backends(args)
    except ValueError as e:
        return _fail(f"score: {e}", 2)
    rows, errors = score_records_with_errors(records, chat, embedder, ctx_k=args.ctx_k)
    write_scores(out, rows, scores_meta(chat, embedder, ctx_k=args.ctx_k,
                                        runs_path=args.store, errors=errors))
    judge_errors = sum(metric != "run" for _, metric, _ in errors)
    print(f"scored {len(rows)} row(s) into {out}; judge errors {judge_errors}, "
          f"run errors {len(errors) - judge_errors}")
    return 0


def add_agreement_parser(sub) -> None:
    p = sub.add_parser("agreement", help="compare two scores files of the same run")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)


def agreement(args) -> int:
    for path in (args.a, args.b):
        if not pathlib.Path(path).is_file():
            return _fail(f"agreement: {path} does not exist", 2)
    print(report.agreement_table(report.judge_agreement(read_scores(args.a),
                                                        read_scores(args.b))) + "\n")
    for label, path in (("a", args.a), ("b", args.b)):
        meta = read_meta(path)
        print(f"{label}: judge {meta.get('judge')}" if meta else f"{label}: provenance: none")
    return 0


COMMANDS = {"run": run, "score": score, "agreement": agreement}
ADD_PARSERS = (add_run_parser, add_score_parser, add_agreement_parser)

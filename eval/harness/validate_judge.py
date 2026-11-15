"""Check the faithfulness judge against RAGBench adherence labels.

RAGBench (galileo-ai/ragbench, CC-BY-4.0) labels each response as adherent to
its documents or not. The judge predicts "adherent" when its faithfulness score
is at least the threshold. Rows where the judge fails (None score,
JudgeParseError, BackendError) are counted in `errors` and left out of every
other metric. Fetching the parquet is the only network access besides the
judge backend.
"""
import argparse
import datetime
import json
import os
import random
import sys
from collections.abc import Sequence
from typing import Callable

from harness.backends import BackendError, CachedChat
from harness.judge_select import (build_judge, judge_args_error, missing_key_env,
                                   parse_extra_body)
from harness.judges import ChatBackend, JudgeParseError, score_faithfulness
from harness.stats import cohens_kappa

_LEGACY_MODEL = "mistral-small-latest"
_REQUIRED = ("id", "question", "documents", "response", "adherence_score")
_PARQUET_URL = ("https://huggingface.co/datasets/galileo-ai/ragbench/resolve/"
                "refs%2Fconvert%2Fparquet/{subset}/{split}/0000.parquet")


def fetch_ragbench(subset: str, split: str):
    """Download one RAGBench parquet split as a dataframe (network)."""
    import pandas as pd
    return pd.read_parquet(_PARQUET_URL.format(subset=subset, split=split))


def load_ragbench_sample(df, n: int, seed: int = 0) -> list[dict]:
    """Deterministic sample of n rows, as balanced between classes as the data allows."""
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe lacks required columns: {missing}")
    rows = [{
        "id": str(r["id"]), "question": r["question"],
        "documents": [str(d) for d in r["documents"]],
        "response": r["response"], "adherent": bool(r["adherence_score"]),
    } for r in df.to_dict("records")]
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(seed)
    pools = {True: [r for r in rows if r["adherent"]],
             False: [r for r in rows if not r["adherent"]]}
    for pool in pools.values():
        rng.shuffle(pool)
    n = min(n, len(rows))
    each = min(n // 2, len(pools[True]), len(pools[False]))
    picked = pools[True][:each] + pools[False][:each]
    rest = pools[True][each:], pools[False][each:]
    larger = rest[0] if len(rest[0]) >= len(rest[1]) else rest[1]
    picked += larger[:n - len(picked)]
    return sorted(picked, key=lambda r: r["id"])


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def evaluate_judge(backend: ChatBackend, sample: Sequence[dict],
                   threshold: float = 0.8) -> dict:
    """Agreement between judge and labels; `n` counts only rows the judge scored."""
    labels: list[bool] = []
    predictions: list[bool] = []
    errors = 0
    for row in sample:
        try:
            score = score_faithfulness(backend, row["response"], row["documents"])
        except (JudgeParseError, BackendError):
            score = None
        if score is None:
            errors += 1
            continue
        labels.append(row["adherent"])
        predictions.append(score >= threshold)
    pairs = list(zip(labels, predictions))
    tp = sum(1 for y, p in pairs if y and p)
    fp = sum(1 for y, p in pairs if not y and p)
    tn = sum(1 for y, p in pairs if not y and not p)
    fn = sum(1 for y, p in pairs if y and not p)
    n = len(pairs)
    return {
        "n": n,
        "accuracy": _ratio(tp + tn, n),
        "kappa": cohens_kappa(labels, predictions) if n else None,
        "precision_adherent": _ratio(tp, tp + fp),
        "recall_adherent": _ratio(tp, tp + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None, *,
         fetch: Callable = fetch_ragbench, backend: ChatBackend | None = None,
         clock: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(
             datetime.timezone.utc), session=None) -> int:
    """Run the CLI. `fetch`, `backend`, `clock` and `session` are test hooks that avoid the network, the key and real time."""
    parser = argparse.ArgumentParser(prog="harness.validate_judge")
    parser.add_argument("--subset", default="techqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--backend", choices=["mistral"], default="mistral")
    parser.add_argument("--judge-model", help="a pinned, dated model id")
    parser.add_argument("--judge-base-url", help="OpenAI-compatible endpoint; default Mistral")
    parser.add_argument("--judge-key-env", help="variable holding the key for --judge-base-url")
    parser.add_argument("--judge-extra-body", help="JSON object merged into every judge request")
    parser.add_argument("--cache")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.out and os.path.exists(args.out):
        print(f"error: {args.out} already exists; refusing to overwrite", file=sys.stderr)
        return 2
    if backend is None:
        problem = judge_args_error(args.judge_base_url, args.judge_key_env)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2
        model = args.judge_model
        if not model:
            if args.judge_base_url:
                print("error: --judge-model is required with --judge-base-url",
                      file=sys.stderr)
                return 2
            model = _LEGACY_MODEL
            print(f"notice: --judge-model not given; using the alias {model}, which is "
                  "not pinned", file=sys.stderr)
        missing = missing_key_env(args.judge_base_url, args.judge_key_env)
        if missing:
            print(f"error: set {missing} in the environment", file=sys.stderr)
            return 2
        try:
            extra = parse_extra_body(args.judge_extra_body)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        backend = build_judge(model, args.judge_base_url, args.judge_key_env,
                              session=session, extra_body=extra)
    if args.cache:
        backend = CachedChat(backend, args.cache)
    sample = load_ragbench_sample(fetch(args.subset, args.split), args.n, args.seed)
    result = evaluate_judge(backend, sample, args.threshold)
    result["params"] = {
        "subset": args.subset, "split": args.split, "n_requested": args.n,
        "n_sample": len(sample), "seed": args.seed, "threshold": args.threshold,
        "backend": backend.name, "cache": bool(args.cache),
        "created_utc": clock().astimezone(datetime.timezone.utc).isoformat(),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "x", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

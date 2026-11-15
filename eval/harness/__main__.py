"""Command line: `python -m harness run|score|agreement|report`."""
import argparse
import sys
from typing import Sequence

from harness import cli_pipeline, report
from harness.metrics import ScoredRow
from harness.run_artifacts import RunStore
from harness.score_runs import read_meta, read_scores


def _coverage_line(rows: Sequence[ScoredRow]) -> str:
    coverage, accuracy = report.coverage_accuracy(rows)
    graded = sum(r.answered and r.answer_correctness is not None for r in rows)
    return (f"coverage {report.fmt(coverage)} (answered / {len(rows)} rows); "
            f"accuracy among answered {report.fmt(accuracy)} "
            f"(correctness >= 4, n={graded})")


def _provenance_line(meta: dict | None) -> str:
    if meta is None:
        return "provenance: none"
    return (f"provenance: judge {meta.get('judge')}, embedder {meta.get('embedder')}, "
            f"ctx_k {meta.get('ctx_k')}, judge errors {len(meta.get('judge_errors', []))}")


def _report(runs: str, scores: str, allow_errors: bool) -> int:
    records = RunStore(runs).load()
    n_errors = sum(r.status == "error" for r in records)
    if n_errors and not allow_errors:
        print(f"{runs}: {n_errors} run error record(s); rerun them with retry_errors "
              "or pass --allow-errors to report without them", file=sys.stderr)
        return 1
    rows = read_scores(scores)
    meta = read_meta(scores)
    ctx_k = meta.get("ctx_k") if meta else None
    print(_provenance_line(meta) + "\n")
    split_of = {r.row_id: r.split for r in records}
    # one table per split (real vs synthetic), never pooled
    for split in sorted({split_of.get(r.row_id, "unknown") for r in rows}):
        part = [r for r in rows if split_of.get(r.row_id, "unknown") == split]
        run_errors = sum(r.status == "error" and r.expected_behavior == "answer"
                         and r.split == split for r in records)
        print(f"## Scores, split: {split}\n")
        print(report.metric_table(part, ctx_k=ctx_k, run_errors=run_errors))
        print("\n" + _coverage_line(part) + "\n")
    print("## Abstention behavior\n")
    print(report.abstention_table(records))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="report on a scored run")
    rep.add_argument("--runs", required=True)
    rep.add_argument("--scores", required=True)
    rep.add_argument("--allow-errors", action="store_true",
                     help="report even if the runs file holds error records")
    for add_parser in cli_pipeline.ADD_PARSERS:
        add_parser(sub)
    args = parser.parse_args(argv)
    if args.command in cli_pipeline.COMMANDS:
        code = cli_pipeline.COMMANDS[args.command](args)
        if args.command == "score" and code == 0:
            print()
            # score already refused error records unless --allow-errors was given.
            return _report(args.store, args.out, allow_errors=True)
        return code
    return _report(args.runs, args.scores, args.allow_errors)


if __name__ == "__main__":
    raise SystemExit(main())

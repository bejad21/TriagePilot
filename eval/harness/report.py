import math
from typing import Sequence

from harness import stats
from harness.metrics import ScoredRow, aggregate, answered_rate
from harness.run_artifacts import RunRecord

# Kappa binarises each metric at its own scale: 0..1 metrics at 0.8, correctness (1..5) at 4.
_AGREEMENT_THRESHOLDS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.8,
    "answer_correctness": 4.0,
    "context_precision": 0.8,
    "context_recall": 0.8,
}
_BASIS_LABEL = {"answered": "answered rows", "all": "all rows"}


def fmt(x: float) -> str:
    """Three decimals, or 'nan'."""
    return "nan" if math.isnan(x) else f"{x:.3f}"


def _ci(lo: float, hi: float) -> str:
    return "n/a" if math.isnan(lo) else f"[{fmt(lo)}, {fmt(hi)}]"


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def metric_table(rows: Sequence[ScoredRow], ctx_k: int | None = None,
                 k_note: bool = True, run_errors: int | None = None) -> str:
    """Markdown table of each metric with its 95% CI, n and denominator basis.

    With `k_note`, a footnote states K for hit_at_k and reciprocal_rank: `ctx_k`
    when known, else the words "first ctx_k chunks". Legacy CSVs pass k_note=False.
    `run_errors`, when given, is printed as the count of runner error rows left out.
    """
    body = []
    if rows:
        rate, lo, hi = answered_rate(rows)
        k = sum(r.answered for r in rows)
        body.append(["answered_rate", fmt(rate), _ci(lo, hi), str(len(rows)),
                     f"all rows ({k} answered)"])
    for s in aggregate(rows).values():
        basis = _BASIS_LABEL[s.basis]
        if s.n_excluded:
            basis += f"; {s.n_excluded} without a value excluded"
        body.append([s.name, fmt(s.mean), _ci(s.lo, s.hi), str(s.n), basis])
    table = _table(["metric", "value", "95% CI", "n", "basis"], body)
    if k_note:
        first = f"first {ctx_k} chunks" if ctx_k is not None else "first ctx_k chunks"
        table += (f"\n\nhit_at_k and reciprocal_rank: K = {first} ranked by reranker score"
                  " (the chunks the judges see)")
    if run_errors is not None:
        table += f"\n\nrun errors excluded: {run_errors}"
    return table


def abstention_table(records: Sequence[RunRecord]) -> str:
    """Behavior table split by expected behavior; escalate rows report false-answer rate.

    Runner error rows are neither answers nor abstentions: they leave every
    denominator and are counted on their own line.
    """
    body = []
    n_errors = sum(r.status == "error" for r in records)
    records = [r for r in records if r.status != "error"]
    escalate = [r for r in records if r.expected_behavior == "escalate"]
    if escalate:
        n = len(escalate)
        basis = "all escalate-expected rows"
        answered = sum(r.status == "answered" for r in escalate)
        lo, hi = stats.wilson_ci(answered, n)
        body.append(["escalate", "false_answer_rate", f"{answered}/{n} = {fmt(answered / n)}",
                     _ci(lo, hi), str(n), basis])
        escalated = sum(r.status == "escalated" for r in escalate)
        body.append(["escalate", "escalation_rate", f"{escalated}/{n} = {fmt(escalated / n)}",
                     "n/a", str(n), basis])
        rejected = sum(r.status == "rejected" for r in escalate)
        body.append(["escalate", "rejected", str(rejected), "n/a", str(n), basis])
    answerable = [r for r in records if r.expected_behavior == "answer"]
    if answerable:
        n = len(answerable)
        for name, status in (("answer_rate", "answered"), ("escalation_rate", "escalated")):
            k = sum(r.status == status for r in answerable)
            lo, hi = stats.wilson_ci(k, n)
            body.append(["answer", name, f"{k}/{n} = {fmt(k / n)}", _ci(lo, hi), str(n),
                         "all answer-expected rows"])
    table = _table(["expected", "metric", "value", "95% CI", "n", "basis"], body)
    return table + f"\n\nrun errors excluded: {n_errors}"


def coverage_accuracy(rows: Sequence[ScoredRow],
                      correctness_threshold: int = 4) -> tuple[float, float]:
    """(coverage, accuracy among answered); rows without correctness leave the accuracy base."""
    answered = [r for r in rows if r.answered]
    coverage = len(answered) / len(rows) if rows else math.nan
    graded = [r.answer_correctness for r in answered if r.answer_correctness is not None]
    if not graded:
        return coverage, math.nan
    return coverage, sum(c >= correctness_threshold for c in graded) / len(graded)


def agreement_table(agreement: dict[str, dict[str, float]]) -> str:
    """Markdown table of a `judge_agreement` result."""
    return _table(["metric", "pearson", "kappa", "threshold", "n"],
                  [[name, fmt(v["pearson"]), fmt(v["kappa"]), str(v["threshold"]),
                    str(int(v["n"]))] for name, v in agreement.items()])


def judge_agreement(a: Sequence[ScoredRow],
                    b: Sequence[ScoredRow]) -> dict[str, dict[str, float]]:
    """Per metric: pearson on raw values, kappa at `threshold` (value >= threshold) and n.

    Only row ids scored in both sets with a value in both are used. Fewer than 2
    pairs, or a constant series, gives a nan pearson rather than an error.
    """
    b_by_id = {r.row_id: r for r in b}
    result = {}
    for name, threshold in _AGREEMENT_THRESHOLDS.items():
        pairs = [(getattr(r, name), getattr(b_by_id[r.row_id], name))
                 for r in a if r.row_id in b_by_id]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 2:
            result[name] = {"pearson": math.nan, "kappa": math.nan,
                            "threshold": threshold, "n": float(len(pairs))}
            continue
        xs, ys = zip(*pairs)
        result[name] = {
            "pearson": stats.pearson(xs, ys),
            "kappa": stats.cohens_kappa([x >= threshold for x in xs], [y >= threshold for y in ys]),
            "threshold": threshold,
            "n": float(len(pairs)),
        }
    return result

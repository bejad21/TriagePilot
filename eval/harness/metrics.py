import math
from dataclasses import dataclass
from typing import Sequence

from harness import stats

_NAN_CI = (math.nan, math.nan)


@dataclass(frozen=True)
class ScoredRow:
    row_id: int
    answered: bool
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    hit_at_k: float | None = None
    reciprocal_rank: float | None = None
    overall_reported: float | None = None


@dataclass(frozen=True)
class MetricSummary:
    name: str
    mean: float
    n: int
    lo: float
    hi: float
    basis: str
    n_excluded: int


def reciprocal_rank(retrieved_sources: Sequence[str], gold: set[str]) -> float | None:
    """1 / rank of the first gold source, 0.0 if none; None without gold."""
    if not gold:
        return None
    for position, source in enumerate(retrieved_sources, start=1):
        if source in gold:
            return 1 / position
    return 0.0


def hit_at_k(retrieved_sources: Sequence[str], gold: set[str]) -> float | None:
    """1.0 if any retrieved source is gold, else 0.0; None without gold."""
    if not gold:
        return None
    return 1.0 if any(source in gold for source in retrieved_sources) else 0.0


def row_overall(row: ScoredRow) -> float | None:
    """Mean of the four RAGAS metrics; None if any is missing."""
    parts = [row.faithfulness, row.answer_relevancy,
             row.context_precision, row.context_recall]
    if any(p is None for p in parts):
        return None
    return sum(parts) / len(parts)


def _overall_value(row: ScoredRow) -> float | None:
    computed = row_overall(row)
    return computed if computed is not None else row.overall_reported


def _summarize(
    name: str, values: list[float | None], basis: str, wilson: bool = False
) -> MetricSummary:
    usable = [v for v in values if v is not None]
    n_excluded = len(values) - len(usable)
    if not usable:
        return MetricSummary(name, math.nan, 0, *_NAN_CI, basis, n_excluded)
    mean = sum(usable) / len(usable)
    if wilson:
        lo, hi = stats.wilson_ci(round(sum(usable)), len(usable))
    elif len(usable) >= 2:
        _, lo, hi = stats.mean_ci(usable)
    else:
        lo, hi = _NAN_CI
    return MetricSummary(name, mean, len(usable), lo, hi, basis, n_excluded)


def aggregate(rows: Sequence[ScoredRow]) -> dict[str, MetricSummary]:
    """Summarize each metric; rows with None for a metric are excluded from it."""
    answered = [r for r in rows if r.answered]
    result = {}
    for name in ("faithfulness", "answer_relevancy", "answer_correctness"):
        result[name] = _summarize(name, [getattr(r, name) for r in answered], "answered")
    result["overall"] = _summarize(
        "overall", [_overall_value(r) for r in answered], "answered"
    )
    for name in ("context_precision", "context_recall", "reciprocal_rank"):
        result[name] = _summarize(name, [getattr(r, name) for r in rows], "all")
    result["hit_at_k"] = _summarize(
        "hit_at_k", [r.hit_at_k for r in rows], "all", wilson=True
    )
    return result


def answered_rate(rows: Sequence[ScoredRow]) -> tuple[float, float, float]:
    """Wilson (rate, lo, hi) of rows that were answered rather than escalated."""
    k = sum(r.answered for r in rows)
    lo, hi = stats.wilson_ci(k, len(rows))
    return k / len(rows), lo, hi



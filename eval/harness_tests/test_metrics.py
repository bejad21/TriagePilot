import math

import pytest

from harness import metrics
from harness.metrics import ScoredRow


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a"], {"z"}) == 0.0
    assert metrics.reciprocal_rank(["a"], set()) is None


def test_hit_at_k():
    assert metrics.hit_at_k(["a", "b"], {"b"}) == 1.0
    assert metrics.hit_at_k(["a"], {"z"}) == 0.0
    assert metrics.hit_at_k(["a"], set()) is None


def test_row_overall_none_when_any_missing():
    full = ScoredRow(1, True, faithfulness=1.0, answer_relevancy=0.5,
                     context_precision=0.5, context_recall=1.0)
    assert metrics.row_overall(full) == 0.75
    partial = ScoredRow(2, True, faithfulness=1.0, answer_relevancy=0.5,
                        context_precision=0.5)
    assert metrics.row_overall(partial) is None


def test_aggregate_exclusion_and_basis():
    rows = [
        ScoredRow(1, True, faithfulness=1.0, context_recall=1.0, hit_at_k=1.0),
        ScoredRow(2, True, faithfulness=0.8, context_recall=0.5, hit_at_k=None),
        ScoredRow(3, True, faithfulness=0.6, context_recall=0.0, hit_at_k=0.0),
        ScoredRow(4, True, faithfulness=None, context_recall=1.0, hit_at_k=1.0),
        ScoredRow(5, False, faithfulness=0.1, context_recall=0.5, hit_at_k=1.0),
    ]
    agg = metrics.aggregate(rows)
    assert agg["hit_at_k"].n == 4
    assert agg["hit_at_k"].n_excluded == 1
    assert agg["hit_at_k"].basis == "all"
    assert agg["hit_at_k"].mean == 0.75
    assert agg["faithfulness"].basis == "answered"
    assert agg["faithfulness"].n == 3
    assert agg["faithfulness"].n_excluded == 1
    assert agg["faithfulness"].mean == pytest.approx(0.8)
    assert agg["context_recall"].basis == "all"
    assert agg["context_recall"].n == 5


def test_aggregate_too_few_values_gives_nan_ci():
    agg = metrics.aggregate([ScoredRow(1, True, faithfulness=0.5)])
    assert agg["faithfulness"].n == 1
    assert math.isnan(agg["faithfulness"].lo) and math.isnan(agg["faithfulness"].hi)
    assert agg["context_recall"].n == 0
    assert math.isnan(agg["context_recall"].mean)


def test_overall_prefers_computed_falls_back_to_reported():
    rows = [
        ScoredRow(1, True, faithfulness=1.0, answer_relevancy=1.0,
                  context_precision=1.0, context_recall=1.0, overall_reported=0.5),
        ScoredRow(2, True, overall_reported=0.0),
        ScoredRow(3, True),
    ]
    agg = metrics.aggregate(rows)
    assert agg["overall"].n == 2
    assert agg["overall"].n_excluded == 1
    assert agg["overall"].mean == pytest.approx(0.5)


def test_answered_rate():
    rows = [ScoredRow(i, i < 3) for i in range(4)]
    rate, lo, hi = metrics.answered_rate(rows)
    assert rate == 0.75
    assert lo < rate < hi

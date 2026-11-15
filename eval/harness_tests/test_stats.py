import pytest

from harness import stats


def test_wilson_ci_known_values():
    lo, hi = stats.wilson_ci(27, 30)
    assert lo == pytest.approx(0.744, abs=0.001)
    assert hi == pytest.approx(0.965, abs=0.001)
    lo, hi = stats.wilson_ci(26, 30)
    assert lo == pytest.approx(0.703, abs=0.001)
    assert hi == pytest.approx(0.947, abs=0.001)


def test_wilson_ci_zero_successes_lower_bound_is_zero():
    assert stats.wilson_ci(0, 10)[0] == 0.0


@pytest.mark.parametrize("k,n", [(1, 0), (-1, 5), (6, 5)])
def test_wilson_ci_rejects_invalid(k, n):
    with pytest.raises(ValueError):
        stats.wilson_ci(k, n)


def test_mean_ci():
    mean, lo, hi = stats.mean_ci([1.0, 1.0, 1.0, 0.0])
    assert mean == 0.75
    assert lo < 0.75 < hi


def test_mean_ci_needs_two_values():
    with pytest.raises(ValueError):
        stats.mean_ci([1.0])


def test_bootstrap_ci_deterministic_per_seed():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert stats.bootstrap_ci(values, seed=3) == stats.bootstrap_ci(values, seed=3)
    assert stats.bootstrap_ci(values, seed=3) != stats.bootstrap_ci(values, seed=4)


def test_paired_bootstrap_diff_identical_inputs():
    assert stats.paired_bootstrap_diff([1, 2, 3], [1, 2, 3]) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_diff_length_mismatch():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([1, 2], [1])


def test_cohens_kappa_perfect_agreement():
    a = [True, False, True, False]
    assert stats.cohens_kappa(a, a) == 1.0


def test_cohens_kappa_constant_equal_raters():
    assert stats.cohens_kappa([True, True], [True, True]) == 1.0


def test_cohens_kappa_disagreement():
    # po = 3/4, pe = 0.5*0.25 + 0.5*0.75 = 0.5, kappa = 0.25 / 0.5
    a = [True, True, False, False]
    b = [True, False, False, False]
    assert stats.cohens_kappa(a, b) == pytest.approx(0.5)


def test_pearson_perfect_correlation():
    assert stats.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

import math
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mean_ci(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("need at least 2 values")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sem = float(scipy_stats.sem(arr))
    half = float(scipy_stats.t.ppf((1 + confidence) / 2, len(arr) - 1)) * sem
    return mean, mean - half, mean + half


def _percentile_interval(
    samples: np.ndarray, confidence: float
) -> tuple[float, float]:
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return float(lo), float(hi)


def _bootstrap_means(
    arr: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    return arr[idx].mean(axis=1)


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = _percentile_interval(_bootstrap_means(arr, n_boot, seed), confidence)
    return float(arr.mean()), lo, hi


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(diff, n_boot, confidence, seed)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    return float(scipy_stats.pearsonr(x, y)[0])

"""Bootstrap resampling over prompts.

Two resamplers, kept separate as the paper's numbers came from both:
resample_sums draws from a random.Random stream, gap_se from a NumPy Generator.
Each consumes its generator, so the order of calls on a shared generator sets
the draws.
"""
import numpy as np


def resample_sums(rows, rng, n_boot=1000):
    """Return the per-column sums of n_boot resamples of rows (tuples of ints).

    Each resample draws len(rows) rows with replacement via rng.choice.
    """
    idxs = list(range(len(rows)))
    sums = []
    for _ in range(n_boot):
        s = [rows[rng.choice(idxs)] for _ in idxs]
        sums.append([sum(col) for col in zip(*s)])
    return sums


def gap_se(c, a, rng, n_boot=1000):
    """Return the bootstrap SE of mean(c) - mean(a) over paired resamples, or nan if empty."""
    c = np.asarray(c, dtype=float)
    a = np.asarray(a, dtype=float)
    n = len(c)
    if n == 0:
        return float("nan")
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gaps[b] = c[idx].mean() - a[idx].mean()
    return gaps.std(ddof=1)

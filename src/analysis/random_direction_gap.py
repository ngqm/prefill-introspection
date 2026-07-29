"""Print the recognition gap under baseline, refusal-direction, and random-direction ablation.

Gaps are on the full 1,085-prompt set (tab:random_ablation). The random column
is the mean gap over the random-direction runs given by --tags, with its
[min, max]. Each tag is the filename suffix of one run written by
scripts.ablate.apply_random_ablation, e.g. "__random_ablated" (seed 42) or
"__random_ablated_2". The default tags are the ten directions the paper
averages; missing runs are skipped. SEs use 1,000 bootstrap resamples.

Usage:
  python -m src.analysis.random_direction_gap
  python -m src.analysis.random_direction_gap --tags __random_ablated,__random_ablated_2,...
"""
import argparse

import numpy as np

from src.analysis.bootstrap import gap_se
from src.analysis.data import claim, load_pooled, models

MODELS = models("Llama-3.1-8B", "Qwen3-8B", "Qwen3-14B", "Gemma-3-12B", "Gemma-3-27B", "Qwen3-32B")
DEFAULT_TAGS = ["__random_ablated"] + [f"__random_ablated_{n}" for n in range(2, 11)]
B = 1000


def load(slug, tag):
    """Return (control, adv) 0/1 claim arrays over the prompts judged in both conditions, sorted by key."""
    ctrl = load_pooled(slug + tag, "static", "control")
    adv = load_pooled(slug + tag, "adv", "adv_prefill")
    keys = sorted(set(ctrl) & set(adv))
    return (np.array([claim(ctrl[k]) for k in keys], dtype=float),
            np.array([claim(adv[k]) for k in keys], dtype=float))


def se(c, a):
    """Return the bootstrap SE (in %) of the gap mean(c) - mean(a), from a fresh seed-42 generator."""
    return gap_se(c, a, np.random.default_rng(seed=42), B) * 100


def mean_se(rand_results):
    """Return the joint bootstrap SE (in %) of the mean gap across random directions.

    Each draw resamples one index set, applies it to every direction's first
    n_min prompts, and averages the per-direction gaps.
    """
    rng = np.random.default_rng(seed=42)
    rand_results = [(c, a) for (c, a) in rand_results if len(c)]
    if not rand_results:
        return float("nan")
    n_min = min(len(c) for c, _ in rand_results)
    cs = np.stack([c[:n_min] for c, _ in rand_results])
    as_ = np.stack([a[:n_min] for _, a in rand_results])
    g = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n_min, size=n_min)
        gaps_i = cs[:, idx].mean(axis=1) - as_[:, idx].mean(axis=1)
        g[i] = gaps_i.mean()
    return g.std(ddof=1) * 100


def main():
    """Print base, refusal-ablated, and averaged random-direction gaps per model."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS),
                    help="Comma-separated random-direction tags (default: the ten paper seeds).")
    args = ap.parse_args()
    rand_tags = [t for t in args.tags.split(",") if t]

    header = (f"{'Model':<14} | "
              f"{'N':>5} {'Base':>7} {'+/-':>5} | "
              f"{'N':>5} {'RefAbl':>7} {'+/-':>5} | "
              f"{f'RandMean(x{len(rand_tags)})':>13} {'+/-':>5} "
              f"{'[min':>7} {'max]':>7}")
    print(header)
    print("-" * len(header))
    for slug, label in MODELS:
        cb, ab = load(slug, "")
        cr, ar = load(slug, "__ablated")
        rand_results = [load(slug, t) for t in rand_tags]

        bg = (cb.mean() - ab.mean()) * 100 if len(cb) else float("nan")
        rg = (cr.mean() - ar.mean()) * 100 if len(cr) else float("nan")
        rand_gaps = [(c.mean() - a.mean()) * 100 for (c, a) in rand_results if len(c)]
        if rand_gaps:
            mean_str = (f"{np.mean(rand_gaps):>+13.1f} {mean_se(rand_results):>5.2f} "
                        f"{min(rand_gaps):>+7.1f} {max(rand_gaps):>+7.1f}")
        else:
            mean_str = f"{'-':>13} {'-':>5} {'-':>7} {'-':>7}"

        print(f"{label:<14} | "
              f"{len(cb):>5d} {bg:>+7.1f} {se(cb, ab):>5.2f} | "
              f"{len(cr):>5d} {rg:>+7.1f} {se(cr, ar):>5.2f} | "
              f"{mean_str}")


if __name__ == "__main__":
    main()

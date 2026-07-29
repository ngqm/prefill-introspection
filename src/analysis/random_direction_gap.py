"""Recognition gap under random-direction ablation, reported as the mean over
N independently sampled random unit directions with the [min, max] range,
matching tab:random_ablation. Bootstrap SE uses 1,000 resamples.

Each direction is a tag on disk written by apply_random_ablation.py, e.g.
"__random_ablated" (seed 42), "__random_ablated_2", "__random_ablated_3".

Scope note: tab:random_ablation in the paper reports the mean over TEN random
directions across six models; this repo ships only the first few direction runs
(and five models), so the defaults reproduce a subset. The published ten-direction
table was assembled from the full run set on the compute cluster.
To reproduce the paper's ten-direction control, generate ten checkpoints with
distinct seeds and variants, run the gen/judge pipeline for each, then pass
their ten tags with --tags. The default three tags reproduce the
three-direction estimate.

Usage:
  python -m src.analysis.random_direction_gap
  python -m src.analysis.random_direction_gap --tags __random_ablated,__random_ablated_2,...
"""
import argparse
import json
from pathlib import Path
import numpy as np

from src.analysis.filters import is_canonical

ROOT = Path(__file__).resolve().parents[2]
RQ = ROOT / "rq1_runs"
DATASETS = ["harmbench", "jailbreakbench", "strongreject", "socialharmbench"]
MODELS = [
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen__Qwen3-8B", "Qwen3-8B"),
    ("Qwen__Qwen3-14B", "Qwen3-14B"),
    ("google__gemma-3-12b-it", "Gemma-3-12B"),
    ("google__gemma-3-27b-it", "Gemma-3-27B"),
]
DEFAULT_TAGS = ["__random_ablated", "__random_ablated_2", "__random_ablated_3"]
B = 1000


def load(safe: str, tag: str):
    ctrl = {}
    adv = {}
    for ds in DATASETS:
        fc = RQ / f"classified_gen_{safe}{tag}_{ds}_f1.jsonl"
        fa = RQ / f"classified_gen_adv_{safe}{tag}_{ds}_f1.jsonl"
        if fc.exists():
            for line in fc.open():
                r = json.loads(line)
                if (r.get("condition") == "control"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    ctrl[(ds, r["prompt_id"])] = int(r["classifier_label"] == "YES")
        if fa.exists():
            for line in fa.open():
                r = json.loads(line)
                if (r.get("condition") == "adv_prefill"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    adv[(ds, r["prompt_id"])] = int(r["classifier_label"] == "YES")
    keys = sorted(set(ctrl) & set(adv))
    return (np.array([ctrl[k] for k in keys], dtype=float),
            np.array([adv[k] for k in keys], dtype=float))


def se(c, a):
    # Fresh seeded generator so the SE is deterministic and independent of how
    # many other SE calls precede it (the table iterates models in any order).
    rng = np.random.default_rng(seed=42)
    n = len(c)
    if n == 0:
        return float("nan")
    g = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        g[i] = c[idx].mean() - a[idx].mean()
    return g.std(ddof=1) * 100


def mean_se(rand_results):
    """Joint bootstrap SE for the mean gap across seeds. Requires all seeds
    indexed by the same prompt set (keys union over loaded keys). Resamples
    indices once per bootstrap draw and averages the resulting per-seed gaps.
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS),
                    help="Comma-separated random-direction tags. Default is the "
                         "three-direction control; pass ten tags to reproduce "
                         "the paper's ten-direction average.")
    args = ap.parse_args()
    rand_tags = [t for t in args.tags.split(",") if t]

    header = (f"{'Model':<14} | "
              f"{'N':>5} {'Base':>7} {'+/-':>5} | "
              f"{'N':>5} {'RefAbl':>7} {'+/-':>5} | "
              f"{f'RandMean(x{len(rand_tags)})':>13} {'+/-':>5} "
              f"{'[min':>7} {'max]':>7}")
    print(header)
    print("-" * len(header))
    for safe, label in MODELS:
        cb, ab = load(safe, "")
        cr, ar = load(safe, "__ablated")
        rand_results = [load(safe, t) for t in rand_tags]

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

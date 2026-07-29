"""Compute bootstrap standard error on the recognition gap per (model, subset).

Reproduces the per-model claim rates and recognition gaps that appear in
the behavior-change appendix table, and adds a bootstrap SE on the gap
for both the behavior-change subset and the complementary no-change
subset. Also computes SE on the full-set base and ablated gaps for the
small-models appendix table.

Usage:
  python -m src.analysis.bootstrap_gap_se
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.analysis.filters import is_canonical, is_behavior_change

ROOT = Path(__file__).resolve().parents[2]
RQ = ROOT / "rq1_runs"

# Display order matches the appendix tables
MODELS = [
    ("meta-llama__Llama-3.2-3B-Instruct",  "Llama-3.2-3B"),
    ("meta-llama__Llama-3.1-8B-Instruct",  "Llama-3.1-8B"),
    ("meta-llama__llama-3.3-70b-instruct", "Llama-3.3-70B"),
    ("Qwen__Qwen3-4B",                     "Qwen3-4B"),
    ("Qwen__Qwen3-8B",                     "Qwen3-8B"),
    ("Qwen__Qwen3-14B",                    "Qwen3-14B"),
    ("Qwen__Qwen3-32B",                    "Qwen3-32B"),
    ("google__gemma-3-4b-it",              "Gemma-3-4B"),
    ("google__gemma-3-12b-it",             "Gemma-3-12B"),
    ("google__gemma-3-27b-it",             "Gemma-3-27B"),
]
DATASETS = ["harmbench", "jailbreakbench", "strongreject", "socialharmbench"]

B = 1000  # bootstrap resamples
RNG = np.random.default_rng(seed=42)


def _collect_rows(path: Path, condition: str, ds: str, target: dict):
    """Index judged rows of one condition by (ds, prompt_id) into target.

    Keeps canonical prompts whose classifier_label is YES or NO. No-op when the
    file is absent.
    """
    if not path.exists():
        return
    for line in path.open():
        r = json.loads(line)
        if (r.get("condition") == condition
                and r.get("classifier_label") in ("YES", "NO")
                and is_canonical(ds, r["prompt_id"])):
            target[(ds, r["prompt_id"])] = r


def load_pairs(safe_model: str, ablated: bool = False):
    """Return list of (claim_ctrl, claim_adv, guard_ctrl, guard_adv, ds, pid)
    over canonical prompts that have both control and adv-prefill rows.

    ablated selects the __ablated run files when True.
    """
    tag = "__ablated" if ablated else ""
    ctrl_rows = {}
    adv_rows = {}
    for ds in DATASETS:
        _collect_rows(RQ / f"classified_gen_{safe_model}{tag}_{ds}_f1.jsonl",
                      "control", ds, ctrl_rows)
        _collect_rows(RQ / f"classified_gen_adv_{safe_model}{tag}_{ds}_f1.jsonl",
                      "adv_prefill", ds, adv_rows)
    pairs = []
    for k in ctrl_rows.keys() & adv_rows.keys():
        c, a = ctrl_rows[k], adv_rows[k]
        pairs.append((
            int(c["classifier_label"] == "YES"),
            int(a["classifier_label"] == "YES"),
            c.get("guard_label"),
            a.get("guard_label"),
            k[0], k[1],
        ))
    return pairs


def bootstrap_gap(claim_c, claim_a, n_boot=B):
    """Bootstrap SE on the gap = mean(claim_c) - mean(claim_a)."""
    claim_c = np.asarray(claim_c, dtype=float)
    claim_a = np.asarray(claim_a, dtype=float)
    n = len(claim_c)
    if n == 0:
        return float("nan")
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        gaps[b] = claim_c[idx].mean() - claim_a[idx].mean()
    return gaps.std(ddof=1)


def gap_and_se(pairs):
    """Return (gap, bootstrap SE) over a pair list, or (nan, nan) if empty.

    gap = mean(claim_ctrl) - mean(claim_adv) using p[0]/p[1] of each pair.
    """
    if not pairs:
        return float("nan"), float("nan")
    cc = [p[0] for p in pairs]
    ca = [p[1] for p in pairs]
    return np.mean(cc) - np.mean(ca), bootstrap_gap(cc, ca)


def main():
    """Print per-model gaps and bootstrap SEs for the appendix tables.

    Covers the behavior-change subset, its complementary no-change subset,
    the full set at baseline, and the full-set ablated gap.
    """
    print(f"{'Model':<14} | {'BC N':>5} {'BC gap':>7} {'BC SE':>7} | "
          f"{'NC N':>5} {'NC gap':>7} {'NC SE':>7} | "
          f"{'Full N':>6} {'Full gap':>9} {'Full SE':>8} | "
          f"{'Abl gap':>8} {'Abl SE':>7}")
    print("-" * 130)
    for safe, label in MODELS:
        pairs = load_pairs(safe, ablated=False)
        n_total = len(pairs)

        bc_pairs = [p for p in pairs if is_behavior_change({"guard_label": p[2]},
                                                            {"guard_label": p[3]})]
        nc_pairs = [p for p in pairs if not is_behavior_change({"guard_label": p[2]},
                                                                {"guard_label": p[3]})
                                       and p[2] is not None and p[3] is not None]
        bc_gap, bc_se = gap_and_se(bc_pairs)
        nc_gap, nc_se = gap_and_se(nc_pairs)
        full_gap, full_se = gap_and_se(pairs)

        abl_pairs = load_pairs(safe, ablated=True)
        abl_gap, abl_se = gap_and_se(abl_pairs)

        print(f"{label:<14} | "
              f"{len(bc_pairs):>5d} {bc_gap*100:>+7.1f} {bc_se*100:>7.2f} | "
              f"{len(nc_pairs):>5d} {nc_gap*100:>+7.1f} {nc_se*100:>7.2f} | "
              f"{n_total:>6d} {full_gap*100:>+9.1f} {full_se*100:>8.2f} | "
              f"{abl_gap*100:>+8.1f} {abl_se*100:>7.2f}")


if __name__ == "__main__":
    main()

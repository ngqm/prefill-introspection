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


def load_pairs(safe_model: str, ablated: bool = False):
    """Return list of (claim_ctrl, claim_adv, guard_ctrl, guard_adv, ds, pid)
    over canonical prompts that have both control and adv-prefill rows."""
    tag = "__ablated" if ablated else ""
    ctrl_rows = {}
    adv_rows = {}
    for ds in DATASETS:
        fc = RQ / f"classified_gen_{safe_model}{tag}_{ds}_f1.jsonl"
        fa = RQ / f"classified_gen_adv_{safe_model}{tag}_{ds}_f1.jsonl"
        if fc.exists():
            for line in fc.open():
                r = json.loads(line)
                if (r.get("condition") == "control"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    ctrl_rows[(ds, r["prompt_id"])] = r
        if fa.exists():
            for line in fa.open():
                r = json.loads(line)
                if (r.get("condition") == "adv_prefill"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    adv_rows[(ds, r["prompt_id"])] = r
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


def main():
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
        # Behavior change subset
        if bc_pairs:
            cc = [p[0] for p in bc_pairs]
            ca = [p[1] for p in bc_pairs]
            bc_gap = np.mean(cc) - np.mean(ca)
            bc_se = bootstrap_gap(cc, ca)
        else:
            bc_gap = bc_se = float("nan")
        # No change subset
        if nc_pairs:
            cc = [p[0] for p in nc_pairs]
            ca = [p[1] for p in nc_pairs]
            nc_gap = np.mean(cc) - np.mean(ca)
            nc_se = bootstrap_gap(cc, ca)
        else:
            nc_gap = nc_se = float("nan")
        # Full set
        cc_all = [p[0] for p in pairs]
        ca_all = [p[1] for p in pairs]
        full_gap = np.mean(cc_all) - np.mean(ca_all) if pairs else float("nan")
        full_se = bootstrap_gap(cc_all, ca_all) if pairs else float("nan")

        # Ablated full-set gap
        abl_pairs = load_pairs(safe, ablated=True)
        if abl_pairs:
            cc_a = [p[0] for p in abl_pairs]
            ca_a = [p[1] for p in abl_pairs]
            abl_gap = np.mean(cc_a) - np.mean(ca_a)
            abl_se = bootstrap_gap(cc_a, ca_a)
        else:
            abl_gap = abl_se = float("nan")

        print(f"{label:<14} | "
              f"{len(bc_pairs):>5d} {bc_gap*100:>+7.1f} {bc_se*100:>7.2f} | "
              f"{len(nc_pairs):>5d} {nc_gap*100:>+7.1f} {nc_se*100:>7.2f} | "
              f"{n_total:>6d} {full_gap*100:>+9.1f} {full_se*100:>8.2f} | "
              f"{abl_gap*100:>+8.1f} {abl_se*100:>7.2f}")


if __name__ == "__main__":
    main()

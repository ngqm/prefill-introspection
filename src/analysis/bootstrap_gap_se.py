"""Print the intention-probe recognition gap with bootstrap SEs for the ten paper models.

Columns: the behavior-change subset and its complementary no-change subset
(tab:behavior_change), the full 1,085-prompt set at baseline, and the full set
under refusal-direction ablation (tab:small_models). SEs use 1,000 resamples.

Usage:
  python -m src.analysis.bootstrap_gap_se
"""
import numpy as np

from core.config import PAPER_MODELS
from src.analysis.bootstrap import gap_se
from src.analysis.data import claim, load_pooled
from src.analysis.filters import is_behavior_change


def load_pairs(slug):
    """Return (claim_ctrl, claim_adv, guard_ctrl, guard_adv) per prompt judged in both conditions."""
    ctrl = load_pooled(slug, "static", "control")
    adv = load_pooled(slug, "adv", "adv_prefill")
    return [(claim(ctrl[k]), claim(adv[k]), ctrl[k].get("guard_label"), adv[k].get("guard_label"))
            for k in ctrl.keys() & adv.keys()]


def gap_and_se(pairs, rng):
    """Return (gap, bootstrap SE) as fractions, or (nan, nan) for no pairs."""
    if not pairs:
        return float("nan"), float("nan")
    cc = [p[0] for p in pairs]
    ca = [p[1] for p in pairs]
    return np.mean(cc) - np.mean(ca), gap_se(cc, ca, rng)


def main():
    """Print the per-model table."""
    rng = np.random.default_rng(seed=42)
    print(f"{'Model':<14} | {'BC N':>5} {'BC gap':>7} {'BC SE':>7} | "
          f"{'NC N':>5} {'NC gap':>7} {'NC SE':>7} | "
          f"{'Full N':>6} {'Full gap':>9} {'Full SE':>8} | "
          f"{'Abl gap':>8} {'Abl SE':>7}")
    print("-" * 130)
    for slug, label in PAPER_MODELS:
        pairs = load_pairs(slug)
        bc_pairs = [p for p in pairs
                    if is_behavior_change({"guard_label": p[2]}, {"guard_label": p[3]})]
        nc_pairs = [p for p in pairs
                    if not is_behavior_change({"guard_label": p[2]}, {"guard_label": p[3]})
                    and p[2] is not None and p[3] is not None]
        bc_gap, bc_se = gap_and_se(bc_pairs, rng)
        nc_gap, nc_se = gap_and_se(nc_pairs, rng)
        full_gap, full_se = gap_and_se(pairs, rng)
        abl_gap, abl_se = gap_and_se(load_pairs(slug + "__ablated"), rng)

        print(f"{label:<14} | "
              f"{len(bc_pairs):>5d} {bc_gap*100:>+7.1f} {bc_se*100:>7.2f} | "
              f"{len(nc_pairs):>5d} {nc_gap*100:>+7.1f} {nc_se*100:>7.2f} | "
              f"{len(pairs):>6d} {full_gap*100:>+9.1f} {full_se*100:>8.2f} | "
              f"{abl_gap*100:>+8.1f} {abl_se*100:>7.2f}")


if __name__ == "__main__":
    main()

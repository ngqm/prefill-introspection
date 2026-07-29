"""Plot the recognition gap before and after refusal-direction ablation (fig:gap_collapse).

One dumbbell per ablated model on the intention probe. Both points use the
prompts in the baseline behavior-change subset; whiskers are 95% bootstrap
percentile CIs over 1,000 resamples.

Writes paper/figures/diagnosticity_gap.{pdf,png}.

Usage:
  python -m src.plotting.plot_gap_collapse
"""
import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.bootstrap import resample_sums
from src.analysis.data import FIG_DIR, POOL_ORDER, claim, models, paired
from src.analysis.filters import is_behavior_change
from src.plotting.style import apply as apply_style, PALETTE

MODELS = models("Llama-3.1-8B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "Gemma-3-12B", "Gemma-3-27B")


def gap_ci(pairs, rng, n_boot=1000):
    """Return (gap, 95% CI low, 95% CI high, n) in % for (control_claim, prefill_claim) pairs."""
    n = len(pairs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    base = sum(c - t for c, t in pairs) / n * 100.0
    samples = [(sc - st) / n * 100.0 for sc, st in resample_sums(pairs, rng, n_boot)]
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return (base, lo, hi, n)


def main():
    """Draw the dumbbell chart and save it."""
    apply_style()
    rng = random.Random(42)
    rows = []
    for slug, label in MODELS:
        changed = [(ds, pid, c, a) for ds, pid, c, a in paired(slug, datasets=POOL_ORDER)
                   if is_behavior_change(c, a)]
        change_set = {(ds, pid) for ds, pid, _, _ in changed}
        base = gap_ci([(claim(c), claim(a)) for _, _, c, a in changed], rng)
        abl = gap_ci([(claim(c), claim(a))
                      for ds, pid, c, a in paired(slug + "__ablated", datasets=POOL_ORDER)
                      if (ds, pid) in change_set], rng)
        rows.append({"label": label,
                     "base": base[0], "base_lo": base[1], "base_hi": base[2],
                     "abl":  abl[0], "abl_lo":  abl[1], "abl_hi":  abl[2]})
        print(f"{label:<13s}  baseline {base[0]:+5.2f} [{base[1]:+5.2f}, {base[2]:+5.2f}] (n={base[3]})   "
              f"ablated {abl[0]:+5.2f} [{abl[1]:+5.2f}, {abl[2]:+5.2f}] (n={abl[3]})")

    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.6, 0.32 * n + 0.7))
    y = np.arange(n)[::-1]

    for i, r in enumerate(rows):
        vb, va = r["base"], r["abl"]
        b_lower, b_upper = vb - r["base_lo"], r["base_hi"] - vb
        a_lower, a_upper = va - r["abl_lo"],  r["abl_hi"]  - va
        ax.plot([vb, va], [y[i], y[i]], color="black", lw=0.9, zorder=2, solid_capstyle="butt")
        ax.errorbar(vb, y[i], xerr=[[b_lower], [b_upper]], fmt="none", ecolor=PALETTE["indigo"],
                    elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=3)
        ax.errorbar(va, y[i], xerr=[[a_lower], [a_upper]], fmt="none", ecolor=PALETTE["rose"],
                    elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=3)
        ax.plot(vb, y[i], marker="o", color=PALETTE["indigo"], markeredgecolor="black",
                markeredgewidth=0.5, markersize=6, zorder=4)
        ax.plot(va, y[i], marker="o", color=PALETTE["rose"], markeredgecolor="black",
                markeredgewidth=0.5, markersize=6, zorder=4)
        # Baseline label sits right of its whisker; the ablated label sits above
        # its marker so a wide left whisker never pushes it into the model names.
        ax.text(vb + b_upper + 0.8, y[i], f"{vb:+.1f}",
                va="center", ha="left", fontsize=8, color="black")
        ax.text(va, y[i] + 0.18, f"{va:+.1f}",
                va="bottom", ha="center", fontsize=8, color="black")

    ax.axvline(0, color="lightgray", lw=0.7, zorder=1)
    ax.grid(False)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xticks([])
    ax.set_xlabel("Recognition gap (%)")
    ax.set_xlim(-16, 100)
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(axis="y", which="both", color="white")

    legend_handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["indigo"],
                      markeredgecolor="black", markeredgewidth=0.5, markersize=6, label="Baseline"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["rose"],
                      markeredgecolor="black", markeredgewidth=0.5, markersize=6, label="Refusal-ablated"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"diagnosticity_gap.{ext}", dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {FIG_DIR}/diagnosticity_gap.pdf  (and .png)")


if __name__ == "__main__":
    main()

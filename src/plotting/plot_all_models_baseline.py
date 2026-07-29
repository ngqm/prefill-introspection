"""Plot control and prefill claim rates for the ten paper models (fig:all_models_baseline).

One dumbbell per model on the behavior-change subset of the intention probe,
pooled over the four benchmarks, with rows sorted by recognition gap.

Writes paper/figures/all_models_baseline.{pdf,png}.

Usage:
  python -m src.plotting.plot_all_models_baseline
"""
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from core.config import PAPER_MODELS
from src.analysis.data import FIG_DIR, claim, paired
from src.analysis.filters import is_behavior_change
from src.plotting.style import apply as apply_style, PALETTE

COLOR = PALETTE["indigo"]


def rates(slug):
    """Return control and prefill claim rates, their gap (in %), and n on the behavior-change subset, or None."""
    pairs = [(claim(c), claim(a)) for _, _, c, a in paired(slug) if is_behavior_change(c, a)]
    if not pairs:
        return None
    n = len(pairs)
    ctrl = sum(c for c, _ in pairs) / n * 100
    pre = sum(a for _, a in pairs) / n * 100
    return {"ctrl": ctrl, "pre": pre, "gap": ctrl - pre, "n": n}


def main():
    """Draw the dumbbell chart and save it."""
    apply_style()
    results = []
    for slug, label in PAPER_MODELS:
        r = rates(slug)
        if r is None:
            print(f"[skip] {label} (missing data)")
            continue
        results.append((label, r))
        print(f"{label:<14s} ctrl={r['ctrl']:5.1f}  pre={r['pre']:5.1f}  gap={r['gap']:+5.1f}  n={r['n']}")
    results.sort(key=lambda x: x[1]["gap"], reverse=True)

    n = len(results)
    fig, ax = plt.subplots(figsize=(3.6, 0.25 * n + 0.6))
    y = np.arange(n)[::-1]

    for i, (label, r) in enumerate(results):
        ax.plot([r["pre"], r["ctrl"]], [y[i], y[i]], color="black", lw=0.7, zorder=2, solid_capstyle="butt")
        ax.plot(r["pre"], y[i], marker="o", color="white", markeredgecolor=COLOR,
                markeredgewidth=1.4, markersize=6, zorder=4)
        ax.plot(r["ctrl"], y[i], marker="o", color=COLOR, markeredgecolor="black",
                markeredgewidth=0.5, markersize=6, zorder=4)
        left_val, right_val = sorted([r["pre"], r["ctrl"]])
        ax.text(left_val - 4, y[i], f"{left_val:.1f}", va="center", ha="right", fontsize=8, color="black")
        ax.text(right_val + 4, y[i], f"{right_val:.1f}", va="center", ha="left", fontsize=8, color="black")

    ax.grid(False)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in results])
    ax.set_xticks([])
    ax.set_xlabel("Introspective claim rate (%)")
    ax.set_xlim(-10, 115)
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(axis="y", which="both", color="white")

    legend_handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR,
                      markeredgecolor="black", markeredgewidth=0.5, markersize=6, label="Control"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor=COLOR, markeredgewidth=1.4, markersize=6, label="Prefilled"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"all_models_baseline.{ext}", dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {FIG_DIR}/all_models_baseline.pdf")


if __name__ == "__main__":
    main()

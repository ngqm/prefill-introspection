"""Plot control-response ASR before and after refusal-direction ablation (fig:asr_validation).

ASR is the share of control responses Llama Guard 3 1B labels unsafe, pooled
over the four benchmarks (1,085 prompts per model).

Writes paper/figures/ablation_asr_validation.{pdf,png}.

Usage:
  python -m src.plotting.plot_ablation_asr
"""
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.data import DATASETS, FIG_DIR, canonical_rows, judged_path, models
from src.plotting.style import apply as apply_style, PALETTE

MODELS = models("Llama-3.2-3B", "Llama-3.1-8B", "Qwen3-4B", "Qwen3-8B",
                "Qwen3-14B", "Gemma-3-12B", "Gemma-3-27B")


def pooled_asr(slug):
    """Return (% unsafe, n) over canonical control responses, or None if a file or guard label is missing."""
    n = u = 0
    for ds in DATASETS:
        p = judged_path(slug, ds)
        if not p.exists():
            return None
        for r in canonical_rows(p, ds):
            if r.get("condition") != "control":
                continue
            gl = r.get("guard_label")
            if gl is None:
                return None
            n += 1
            if gl == "unsafe":
                u += 1
    return (u / n * 100, n) if n else None


def main():
    """Draw the grouped bar chart and save it."""
    apply_style()
    baseline, ablated, labels = [], [], []
    for slug, label in MODELS:
        b = pooled_asr(slug)
        a = pooled_asr(slug + "__ablated")
        assert b is not None and a is not None, f"missing data for {label}"
        baseline.append(b[0])
        ablated.append(a[0])
        labels.append(label)
        print(f"{label:<14s}  baseline={b[0]:5.1f}({b[1]})  ablated={a[0]:5.1f}({a[1]})")

    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    x = np.arange(len(labels))
    w = 0.30
    b1 = ax.bar(x - w/2, baseline, w, label="Baseline",
                color=PALETTE["cyan"], edgecolor="none", linewidth=0)
    b2 = ax.bar(x + w/2, ablated,  w, label="Refusal-ablated",
                color=PALETTE["rose"], edgecolor="none", linewidth=0)

    for bars, vals in ((b1, baseline), (b2, ablated)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "ablation_asr_validation.pdf"
    out_png = FIG_DIR / "ablation_asr_validation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"[ok] wrote {out_pdf}  and {out_png}")


if __name__ == "__main__":
    main()

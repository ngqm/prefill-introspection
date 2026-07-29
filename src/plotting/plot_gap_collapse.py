"""RQ3 gap-collapse figure (fig:gap_collapse in the paper): the intention-probe
recognition gap at baseline vs under refusal-direction ablation, one dumbbell
per ablated model, with 95% bootstrap percentile CIs as horizontal whiskers.

Writes paper/figures/diagnosticity_gap.{pdf,png}, the copy the paper compiles
against.
"""
import json, random, statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.plotting.style import apply as apply_style, PALETTE
from src.analysis.filters import is_canonical, is_behavior_change

apply_style()
random.seed(42)

RQ = Path("rq1_runs")
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ["HarmBench", "SocialHarmBench", "JailbreakBench", "StrongREJECT"]


def _read(safe, ds, ftag, adv):
    p = RQ / f"classified_gen{'_adv' if adv else ''}_{safe}_{ds.lower()}_{ftag}.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.open():
        r = json.loads(line)
        if not is_canonical(ds, r["prompt_id"]):
            continue
        rows.append(r)
    return rows


def paired(safe, ftag="f1", change_set=None):
    """Behavior-change subset on baseline pooled across DATASETS.
    If change_set (dict ds -> set of prompt_ids) is given, restrict to
    that prompt set (used to evaluate the ablated checkpoint on the same
    prompts that flipped at baseline)."""
    triples = []
    chg_lookup = {}
    for ds in DATASETS:
        static_rows = _read(safe, ds, ftag, adv=False)
        adv_rows = _read(safe, ds, ftag, adv=True)
        if static_rows is None or adv_rows is None:
            continue
        ctrl = {r["prompt_id"]: r for r in static_rows if r.get("condition") == "control"}
        adv = {r["prompt_id"]: r for r in adv_rows}
        ds_chg = set()
        for pid in set(ctrl) & set(adv):
            if change_set is not None:
                if pid not in change_set.get(ds, set()):
                    continue
            else:
                if not is_behavior_change(ctrl[pid], adv[pid]):
                    continue
            triples.append((pid,
                            int(ctrl[pid].get("classifier_label") == "YES"),
                            int(adv[pid].get("classifier_label") == "YES")))
            ds_chg.add(pid)
        chg_lookup[ds] = ds_chg
    return triples, chg_lookup


def gap_ci(triples, n_boot=1000):
    """Return (point estimate, 95% lower CI, 95% upper CI, n) using
    bootstrap percentile method."""
    n = len(triples)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    base = sum(c - t for _, c, t in triples) / n * 100.0
    samples = []
    indices = list(range(n))
    for _ in range(n_boot):
        s = [triples[random.choice(indices)] for _ in indices]
        samples.append(sum(c - t for _, c, t in s) / n * 100.0)
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return (base, lo, hi, n)


PAIRS = [
    ("Llama-3.1-8B",  "meta-llama__Llama-3.1-8B-Instruct"),
    ("Qwen3-8B",      "Qwen__Qwen3-8B"),
    ("Qwen3-14B",     "Qwen__Qwen3-14B"),
    ("Gemma-3-12B",   "google__gemma-3-12b-it"),
    ("Gemma-3-27B",   "google__gemma-3-27b-it"),
    ("Qwen3-32B",     "Qwen__Qwen3-32B"),
]

rows = []
for label, safe in PAIRS:
    base_triples, change_set = paired(safe)
    abl_triples, _ = paired(safe + "__ablated", change_set=change_set)
    b = gap_ci(base_triples)
    a = gap_ci(abl_triples)
    rows.append({"label": label,
                 "base": b[0], "base_lo": b[1], "base_hi": b[2],
                 "abl":  a[0], "abl_lo":  a[1], "abl_hi":  a[2]})
    print(f"{label:<13s}  baseline {b[0]:+5.2f} [{b[1]:+5.2f}, {b[2]:+5.2f}] (n={b[3]})   "
          f"ablated {a[0]:+5.2f} [{a[1]:+5.2f}, {a[2]:+5.2f}] (n={a[3]})")


n = len(rows)
fig, ax = plt.subplots(figsize=(3.6, 0.32 * n + 0.7))
y = np.arange(n)[::-1]

for i, r in enumerate(rows):
    vb, va = r["base"], r["abl"]
    # Asymmetric 95% CI whisker half-widths.
    b_lower, b_upper = vb - r["base_lo"], r["base_hi"] - vb
    a_lower, a_upper = va - r["abl_lo"],  r["abl_hi"]  - va
    # Connecting line first so markers sit on top.
    ax.plot([vb, va], [y[i], y[i]], color="black", lw=0.9, zorder=2, solid_capstyle="butt")
    # 95% CI whiskers, drawn before markers so they sit underneath.
    ax.errorbar(vb, y[i], xerr=[[b_lower], [b_upper]], fmt="none", ecolor=PALETTE["slate"],
                elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=3)
    ax.errorbar(va, y[i], xerr=[[a_lower], [a_upper]], fmt="none", ecolor=PALETTE["terracotta"],
                elinewidth=1.0, capsize=2.4, capthick=1.0, zorder=3)
    # Markers on top.
    ax.plot(vb, y[i], marker="o", color=PALETTE["slate"], markeredgecolor="black",
            markeredgewidth=0.5, markersize=6, zorder=4)
    ax.plot(va, y[i], marker="o", color=PALETTE["terracotta"], markeredgecolor="black",
            markeredgewidth=0.5, markersize=6, zorder=4)
    # Baseline value label: right of baseline marker, outside the whisker.
    ax.text(vb + b_upper + 0.8, y[i], f"{vb:+.1f}",
            va="center", ha="left", fontsize=8, color=PALETTE["slate"])
    # Ablated value label: centered above the ablated marker, so a wide left
    # whisker never pushes it into the y-axis model names.
    ax.text(va, y[i] + 0.18, f"{va:+.1f}",
            va="bottom", ha="center", fontsize=8, color=PALETTE["terracotta"])

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

import matplotlib.lines as mlines
legend_handles = [
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["slate"],
                  markeredgecolor="black", markeredgewidth=0.5, markersize=6, label="Baseline"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["terracotta"],
                  markeredgecolor="black", markeredgewidth=0.5, markersize=6, label="Refusal-ablated"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=False, handletextpad=0.4, columnspacing=1.2)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"diagnosticity_gap.{ext}", dpi=300, bbox_inches="tight")
print(f"[ok] wrote {OUT_DIR}/diagnosticity_gap.pdf  (and .png)")

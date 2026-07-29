"""All-ten-models baseline RQ1 figure: F1 claim rate at control vs adv-prefill,
dumbbell-style. One row per model, grouped by family. Replaces the
appendix table tab:all_models_f1 in the main text.
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


def _read(safe, ds, adv):
    p = RQ / f"classified_gen{'_adv' if adv else ''}_{safe}_{ds.lower()}_f1.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.open():
        r = json.loads(line)
        if not is_canonical(ds, r["prompt_id"]):
            continue
        rows.append(r)
    return rows


def paired(safe):
    """Behavior-change subset: rows where guard label differs between
    control and adv_prefill conditions. Pooled across DATASETS."""
    triples = []
    for ds in DATASETS:
        static_rows = _read(safe, ds, adv=False)
        adv_rows = _read(safe, ds, adv=True)
        if static_rows is None or adv_rows is None:
            continue
        ctrl = {r["prompt_id"]: r for r in static_rows if r.get("condition") == "control"}
        adv = {r["prompt_id"]: r for r in adv_rows}
        for pid in set(ctrl) & set(adv):
            if not is_behavior_change(ctrl[pid], adv[pid]):
                continue
            triples.append((pid,
                            int(ctrl[pid].get("classifier_label") == "YES"),
                            int(adv[pid].get("classifier_label") == "YES")))
    return triples


def rates(triples):
    if not triples:
        return None
    n = len(triples)
    ctrl = sum(c for _, c, _ in triples) / n * 100
    pre = sum(a for _, _, a in triples) / n * 100
    return {"ctrl": ctrl, "pre": pre, "gap": ctrl - pre, "n": n}


# Display order: by family, smallest to largest
MODELS = [
    ("Llama-3.2-3B",  "meta-llama__Llama-3.2-3B-Instruct",  "llama"),
    ("Llama-3.1-8B",  "meta-llama__Llama-3.1-8B-Instruct",  "llama"),
    ("Llama-3.3-70B", "meta-llama__llama-3.3-70b-instruct", "llama"),
    ("Qwen3-4B",      "Qwen__Qwen3-4B",                     "qwen"),
    ("Qwen3-8B",      "Qwen__Qwen3-8B",                     "qwen"),
    ("Qwen3-14B",     "Qwen__Qwen3-14B",                    "qwen"),
    ("Qwen3-32B",     "Qwen__Qwen3-32B",                    "qwen"),
    ("Gemma-3-4B",    "google__gemma-3-4b-it",              "gemma"),
    ("Gemma-3-12B",   "google__gemma-3-12b-it",             "gemma"),
    ("Gemma-3-27B",   "google__gemma-3-27b-it",             "gemma"),
]

COLOR = PALETTE["slate"]

results = []
for label, safe, fam in MODELS:
    r = rates(paired(safe))
    if r is None:
        print(f"[skip] {label} (missing data)")
        continue
    results.append((label, safe, fam, r))
    print(f"{label:<14s} ctrl={r['ctrl']:5.1f}  pre={r['pre']:5.1f}  gap={r['gap']:+5.1f}  n={r['n']}")

# Sort by recognition gap (largest gap at the top)
results.sort(key=lambda x: x[3]["gap"], reverse=True)

n = len(results)
fig, ax = plt.subplots(figsize=(3.6, 0.25 * n + 0.6))
y = np.arange(n)[::-1]

for i, (label, safe, fam, r) in enumerate(results):
    # Connecting line (gap)
    ax.plot([r["pre"], r["ctrl"]], [y[i], y[i]], color="black", lw=0.7, zorder=2, solid_capstyle="butt")
    # Prefill marker (hollow)
    ax.plot(r["pre"], y[i], marker="o", color="white", markeredgecolor=COLOR,
            markeredgewidth=1.4, markersize=6, zorder=4)
    # Control marker (filled)
    ax.plot(r["ctrl"], y[i], marker="o", color=COLOR, markeredgecolor="black",
            markeredgewidth=0.5, markersize=6, zorder=4)
    # Annotate each marker with its claim rate value.
    # Place prefill label to the left of its marker, control label to the right.
    left_val, right_val = sorted([r["pre"], r["ctrl"]])
    left_label = f"{r['pre']:.1f}" if r["pre"] < r["ctrl"] else f"{r['ctrl']:.1f}"
    right_label = f"{r['ctrl']:.1f}" if r["pre"] < r["ctrl"] else f"{r['pre']:.1f}"
    ax.text(left_val - 4, y[i], left_label, va="center", ha="right", fontsize=8, color=COLOR)
    ax.text(right_val + 4, y[i], right_label, va="center", ha="left", fontsize=8, color=COLOR)

ax.grid(False)
ax.set_yticks(y)
ax.set_yticklabels([label for label, _, _, _ in results])
ax.set_xticks([])
ax.set_xlabel("Introspective claim rate (%)")
ax.set_xlim(-10, 115)
for s in ax.spines.values():
    s.set_color("white")
ax.tick_params(axis="y", which="both", color="white")

import matplotlib.lines as mlines
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
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"all_models_baseline.{ext}", dpi=300, bbox_inches="tight")
print(f"[ok] wrote {OUT_DIR}/all_models_baseline.pdf")

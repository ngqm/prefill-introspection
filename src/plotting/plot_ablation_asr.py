"""Bar chart for the appendix: baseline vs ablated ASR averaged across the
four safety benchmarks (N=1,085 prompts per model per condition). Used to
justify that the refusal-direction ablation is materially active before we
attribute recognition-gap changes to it.
"""
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.plotting.style import apply as apply_style, PALETTE

apply_style()
plt.rcParams["hatch.color"] = "white"
plt.rcParams["hatch.linewidth"] = 0.8

OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RQ = Path("rq1_runs")

with open("data/harmbench_behaviors_text_test.csv") as f:
    HB_STANDARD = {r["BehaviorID"] for r in csv.DictReader(f) if r["FunctionalCategory"] == "standard"}

DATASETS = ["harmbench", "socialharmbench", "jailbreakbench", "strongreject"]


def pooled_asr(safe, datasets=DATASETS):
    """ASR on the natural (control) response, pooled across the given
    datasets. Returns (pct_unsafe, N) or None if any required dataset file
    is missing guard labels."""
    n = u = 0
    for ds in datasets:
        p = RQ / f"classified_gen_{safe}_{ds}_f1.jsonl"
        if not p.exists():
            return None
        for line in p.open():
            r = json.loads(line)
            if r.get("condition") != "control":
                continue
            if ds == "harmbench" and r["prompt_id"] not in HB_STANDARD:
                continue
            gl = r.get("guard_label")
            if gl is None:
                return None  # no guard data for this row
            n += 1
            if gl == "unsafe":
                u += 1
    return (u / n * 100, n) if n else None


ABLATED = [
    ("meta-llama__Llama-3.2-3B-Instruct", "Llama-3.2-3B"),
    ("meta-llama__Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen__Qwen3-4B",                    "Qwen3-4B"),
    ("Qwen__Qwen3-8B",                    "Qwen3-8B"),
    ("Qwen__Qwen3-14B",                   "Qwen3-14B"),
    ("google__gemma-3-12b-it",            "Gemma-3-12B"),
    ("google__gemma-3-27b-it",            "Gemma-3-27B"),
]

baseline, ablated, labels = [], [], []
for safe, label in ABLATED:
    b = pooled_asr(safe)
    a = pooled_asr(safe + "__ablated")
    assert b is not None and a is not None, f"missing data for {label}"
    baseline.append(b[0])
    ablated.append(a[0])
    labels.append(label)
    print(f"{label:<14s}  baseline={b[0]:5.1f}({b[1]})  ablated={a[0]:5.1f}({a[1]})")

fig, ax = plt.subplots(figsize=(4.5, 2.0))
x = np.arange(len(labels))
w = 0.30
b1 = ax.bar(x - w/2, baseline, w, label="Baseline",
            color=PALETTE["slate_lite"], edgecolor="none", linewidth=0)
b2 = ax.bar(x + w/2, ablated,  w, label="Refusal-ablated",
            color=PALETTE["terracotta"], edgecolor="white", linewidth=0,
            hatch="///")
# Force the hatch color to white on each ablated bar patch.
for patch in b2:
    patch._hatch_color = (1.0, 1.0, 1.0, 1.0)

for bar, val in zip(b1, baseline):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}",
            ha="center", va="bottom", fontsize=7)
for bar, val in zip(b2, ablated):
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
out_pdf = OUT_DIR / "ablation_asr_validation.pdf"
out_png = OUT_DIR / "ablation_asr_validation.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"[ok] wrote {out_pdf}  and {out_png}")
print("Baseline ASR:", dict(zip(labels, baseline)))
print("Ablated  ASR:", dict(zip(labels, ablated)))

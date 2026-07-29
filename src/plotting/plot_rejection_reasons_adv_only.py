"""Per-model rejection-style stacked bars for the adversarial-prefill taxonomy.

Reads taxonomy_reasons/classifications.jsonl (intention-probe rejections under
the adversarial prefill condition across the ten base models, assigned to the
eight author-defined categories by scripts.classify.label_rejection_taxonomy)
and renders one horizontal stacked bar per model.

Output: paper/figures/rejection_reasons_overall.pdf
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.plotting.style import apply as apply_style
from src.analysis.filters import is_canonical
apply_style()


ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATIONS = ROOT / "taxonomy_reasons" / "classifications.jsonl"
OUT = ROOT / "paper" / "figures" / "rejection_reasons_overall.pdf"

# Ten base models, dropping archived gemma-2-9b.
MODELS_KEEP = {
    "meta-llama__Llama-3.2-3B-Instruct",
    "meta-llama__Llama-3.1-8B-Instruct",
    "meta-llama__llama-3.3-70b-instruct",
    "Qwen__Qwen3-4B",
    "Qwen__Qwen3-8B",
    "Qwen__Qwen3-14B",
    "qwen__qwen3-32b",
    "google__gemma-3-4b-it",
    "google__gemma-3-12b-it",
    "google__gemma-3-27b-it",
}

# Order: safety-cited first, then intent-reframing, then error/mistake, then misc.
CATEGORIES = [
    "ethical_policy_violation",
    "safety_protocol_failure",
    "lack_of_intent",
    "hypothetical_framing",
    "poor_phrasing",
    "factual_inaccuracy",
    "training_data_artifact",
    "misinterpretation_of_request",
    "other",
]
COLORS = {
    "ethical_policy_violation":     "#8B3A30",  # deep brick
    "safety_protocol_failure":      "#E5A584",  # light terracotta
    "lack_of_intent":               "#2A3540",  # deep slate
    "hypothetical_framing":         "#9BAAB1",  # light slate
    "poor_phrasing":                "#A8C0A0",  # light sage
    "factual_inaccuracy":           "#2E4A3C",  # deep forest
    "training_data_artifact":       "#C99E5B",  # warm amber
    "misinterpretation_of_request": "#9A6FAD",  # dusty violet
    "other":                        "#D9D5CE",  # warm grey
}
LABEL = {
    "ethical_policy_violation":     "Ethical policy violation",
    "safety_protocol_failure":      "Safety protocol failure",
    "lack_of_intent":               "Lack of intent",
    "hypothetical_framing":         "Hypothetical framing",
    "poor_phrasing":                "Poor phrasing",
    "factual_inaccuracy":           "Factual inaccuracy",
    "training_data_artifact":       "Training data artifact",
    "misinterpretation_of_request": "Misinterpretation of request",
    "other":                        "Other",
}
LIGHT_FILL = {"safety_protocol_failure", "hypothetical_framing", "poor_phrasing",
              "training_data_artifact", "other"}


def _clean_label(model_safe):
    return model_safe.split("__")[-1].lower().replace("-instruct", "").replace("-it", "")


def main():
    per_model = defaultdict(Counter)
    for line in CLASSIFICATIONS.open():
        r = json.loads(line)
        if r["model"] not in MODELS_KEEP:
            continue
        # Restrict to the canonical 1,085-prompt set (HarmBench-standard only);
        # the taxonomy input was built over the full HarmBench behavior set.
        prompt_id = r["id"].split("|")[1] if "|" in r["id"] else r["id"]
        if not is_canonical(r["dataset"], prompt_id):
            continue
        per_model[r["model"]][r.get("category", "other")] += 1

    rows = []
    for m, c in per_model.items():
        n = sum(c.values())
        rows.append({"model": m, "n": n,
                     "pct": {cat: 100.0 * c.get(cat, 0) / n for cat in CATEGORIES}})
    # Sort by share of ethical_policy_violation so families cluster visually.
    rows.sort(key=lambda r: -r["pct"]["ethical_policy_violation"])

    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(rows) + 1.1))
    y = np.arange(len(rows))[::-1]
    labels = [f"{_clean_label(r['model'])}  (n={r['n']})" for r in rows]

    for i, r in enumerate(rows):
        left = 0.0
        for cat in CATEGORIES:
            v = r["pct"][cat]
            if v <= 0: continue
            ax.barh(y[i], v, left=left, color=COLORS[cat],
                    edgecolor="white", linewidth=0.6, height=0.62)
            if v >= 5.0:
                ax.text(left + v / 2, y[i], f"{v:.0f}",
                        va="center", ha="center", fontsize=7,
                        color="#2b2b2b" if cat in LIGHT_FILL else "white")
            left += v

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of rejections", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)

    legend_handles = [mpatches.Patch(color=COLORS[c], label=LABEL[c]) for c in CATEGORIES]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=8, frameon=False, handletextpad=0.5, columnspacing=1.5)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()

"""Plot each model's rejection reasons as one stacked bar (fig:rejection_styles).

Reads taxonomy_reasons/classifications.jsonl, the intention-probe rejections
under adversarial prefill that scripts.classify.label_rejection_taxonomy
assigns to the taxonomy categories, and keeps the ten paper models on the
canonical prompt set.

Writes paper/figures/rejection_reasons_overall.pdf.

Usage:
  python -m src.plotting.plot_rejection_reasons
"""
import json
from collections import Counter, defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from core.config import PAPER_MODELS, ROOT_DIR
from src.analysis.data import FIG_DIR
from src.analysis.filters import is_canonical
from src.plotting.style import apply as apply_style

CLASSIFICATIONS = ROOT_DIR / "taxonomy_reasons" / "classifications.jsonl"
OUT = FIG_DIR / "rejection_reasons_overall.pdf"
DISPLAY = dict(PAPER_MODELS)

# Safety-cited first, then intent reframing, then error or mistake, then the rest.
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
# Paul Tol 'muted'. Related categories share a hue pair (wine/rose, indigo/cyan,
# teal/green); the fixed order and in-segment numbers also identify segments.
COLORS = {
    "ethical_policy_violation":     "#882255",  # wine
    "safety_protocol_failure":      "#CC6677",  # rose
    "lack_of_intent":               "#332288",  # indigo
    "hypothetical_framing":         "#88CCEE",  # cyan
    "poor_phrasing":                "#44AA99",  # teal
    "factual_inaccuracy":           "#117733",  # green
    "training_data_artifact":       "#DDCC77",  # sand
    "misinterpretation_of_request": "#AA4499",  # purple
    "other":                        "#BBBBBB",  # grey
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
# Light fills take dark text; the rest take white text.
LIGHT_FILL = {"hypothetical_framing", "poor_phrasing",
              "training_data_artifact", "other"}


def main():
    """Tally each model's rejection categories and save the stacked bar chart."""
    apply_style()
    per_model = defaultdict(Counter)
    with CLASSIFICATIONS.open() as f:
        for line in f:
            r = json.loads(line)
            if r["model"] not in DISPLAY:
                continue
            prompt_id = r["id"].split("|")[1] if "|" in r["id"] else r["id"]
            if not is_canonical(r["dataset"], prompt_id):
                continue
            per_model[r["model"]][r.get("category", "other")] += 1

    rows = []
    for m, c in per_model.items():
        n = sum(c.values())
        rows.append({"model": m, "n": n,
                     "pct": {cat: 100.0 * c.get(cat, 0) / n for cat in CATEGORIES}})
    rows.sort(key=lambda r: -r["pct"]["ethical_policy_violation"])

    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(rows) + 1.1))
    y = np.arange(len(rows))[::-1]
    labels = [f"{DISPLAY[r['model']]}  (n={r['n']})" for r in rows]

    for i, r in enumerate(rows):
        left = 0.0
        for cat in CATEGORIES:
            v = r["pct"][cat]
            if v <= 0:
                continue
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

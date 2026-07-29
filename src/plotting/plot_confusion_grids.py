"""Plot 2x5 grids of per-model 2x2 contingency tables, one figure per pairing.

The four pairings, selected with --pair (default: all):
  refusal_intention          control refusal vs intention-probe verdict on the
                             prefilled response (fig:refusal_intent_confusion)
  refusal_tampering          control refusal vs tampering-probe verdict
                             (fig:refusal_tamper_confusion)
  prefill_refusal_intention  safety of the prefilled response vs intention-probe
                             verdict on it (fig:prefill_intent_confusion)
  intention_tampering        intention vs tampering verdicts on the same
                             prefilled response (fig:intent_tamper_confusion)

On the tampering probe YES means the model says it was tampered with, so YES
counts as reject. All counts use the canonical prompt set.

Writes paper/figures/<name>_confusion.pdf (prefill_refusal_intention also
writes a .png).

Usage:
  python -m src.plotting.plot_confusion_grids [--pair NAME]
"""
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from core.config import PAPER_MODELS
from src.analysis.data import DATASETS, FIG_DIR, canonical_rows, is_judged, judged_path, load_pooled
from src.plotting.style import apply


def _guard(row):
    """Return the lowercased guard label, or "" if absent."""
    return (row.get("guard_label") or "").lower()


def load_control_refusal_vs_probe(slug, probe, claim_label):
    """Count control refusal against the probe verdict on the prefilled response.

    The control response counts as refuse when Llama Guard 3 1B labels it safe.
    The probe verdict counts as claim when classifier_label equals claim_label.
    Returns (counts, n).
    """
    control_refuse = {}
    for ds in DATASETS:
        for r in canonical_rows(judged_path(slug, ds, "static", "f1"), ds):
            if r.get("condition") == "control" and _guard(r) in ("safe", "unsafe"):
                control_refuse[(ds, r["prompt_id"])] = "refuse" if _guard(r) == "safe" else "comply"
    adv_claim = {k: "claim" if r["classifier_label"] == claim_label else "reject"
                 for k, r in load_pooled(slug, "adv", "adv_prefill", probe).items()}
    common = set(control_refuse) & set(adv_claim)
    counts = defaultdict(int)
    for k in common:
        counts[(control_refuse[k], adv_claim[k])] += 1
    return counts, len(common)


def load_prefill_safety_vs_intention(slug):
    """Count prefilled-response safety against the intention verdict on the same row.

    The first row per prompt counts. Returns (counts, n).
    """
    counts = defaultdict(int)
    seen = set()
    for ds in DATASETS:
        for r in canonical_rows(judged_path(slug, ds, "adv", "f1"), ds):
            if r.get("condition") != "adv_prefill":
                continue
            g = _guard(r)
            if g not in ("safe", "unsafe") or not is_judged(r):
                continue
            key = (ds, r["prompt_id"])
            if key in seen:
                continue
            seen.add(key)
            col = "claim" if r["classifier_label"] == "YES" else "reject"
            counts[(g, col)] += 1
    return counts, len(seen)


def load_intention_vs_tampering(slug):
    """Count the intention verdict against the tampering verdict on the same prefilled response.

    Returns (counts, n).
    """
    intent = load_pooled(slug, "adv", "adv_prefill", "f1")
    tamper = load_pooled(slug, "adv", "adv_prefill", "f2")
    common = set(intent) & set(tamper)
    counts = defaultdict(int)
    for k in common:
        i = "claim" if intent[k]["classifier_label"] == "YES" else "reject"
        t = "reject" if tamper[k]["classifier_label"] == "YES" else "claim"
        counts[(i, t)] += 1
    return counts, len(common)


PAIRINGS = {
    "refusal_intention": dict(
        out="refusal_vs_intention_confusion.pdf",
        rows=["refuse", "comply"], cols=["claim", "reject"],
        xlabel="intention", ylabel="control refusal", png=False,
        load=lambda slug: load_control_refusal_vs_probe(slug, "f1", "YES"),
    ),
    "refusal_tampering": dict(
        out="refusal_vs_tampering_confusion.pdf",
        rows=["refuse", "comply"], cols=["claim", "reject"],
        xlabel="tampering", ylabel="control refusal", png=False,
        load=lambda slug: load_control_refusal_vs_probe(slug, "f2", "NO"),
    ),
    "prefill_refusal_intention": dict(
        out="prefill_refusal_vs_intention_confusion.pdf",
        rows=["unsafe", "safe"], cols=["claim", "reject"],
        xlabel="intention", ylabel="prefill safety", png=True,
        load=load_prefill_safety_vs_intention,
    ),
    "intention_tampering": dict(
        out="intention_vs_tampering_confusion.pdf",
        rows=["claim", "reject"], cols=["claim", "reject"],
        xlabel="tampering", ylabel="intention", png=False,
        load=load_intention_vs_tampering,
    ),
}


def draw(pairing):
    """Render the 2x5 grid for one PAIRINGS entry and save it under FIG_DIR."""
    cfg = PAIRINGS[pairing]
    out = FIG_DIR / cfg["out"]
    rows, cols = cfg["rows"], cfg["cols"]

    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.5))
    for ax, (slug, label) in zip(axes.ravel(), PAPER_MODELS):
        counts, n = cfg["load"](slug)
        mat = np.array([[counts[(r, c)] for c in cols] for r in rows], dtype=float)
        total = mat.sum()
        pct = (mat / total * 100) if total else mat

        ax.imshow(pct, cmap=plt.get_cmap("Greys"), vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                v = pct[i, j]
                ax.text(j, i, f"{int(mat[i, j])}\n({v:.1f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if v > 55 else "black")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(cols, fontsize=8)
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_xlabel(cfg["xlabel"], fontsize=8)
        ax.set_ylabel(cfg["ylabel"], fontsize=8)
        ax.set_title(f"{label} (n={n})", fontsize=9, pad=4)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    if cfg["png"]:
        plt.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] wrote {out}" + (f" and {out.with_suffix('.png')}" if cfg["png"] else ""))


def main():
    """Draw the requested pairing, or all four."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=list(PAIRINGS), default=None,
                        help="Which pairing to draw (default: all four)")
    args = parser.parse_args()
    apply()
    for p in ([args.pair] if args.pair else list(PAIRINGS)):
        draw(p)


if __name__ == "__main__":
    main()

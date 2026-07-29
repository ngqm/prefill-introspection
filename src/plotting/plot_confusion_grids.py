"""2x5 grids of per-model 2x2 confusion matrices, one figure per pairing.

The four pairings, selected with --pair (default: all):
  refusal_intention          control refusal vs intention-probe verdict on the
                             adv_prefill response
  refusal_tampering          control refusal vs tampering-probe verdict
                             (f2 polarity: YES = "was tampered" = reject)
  prefill_refusal_intention  safety of the prefilled response vs intention-probe
                             verdict on the same response
  intention_tampering        intention vs tampering verdicts on the same
                             adv_prefill response

All restricted to the canonical prompt set (src.analysis.filters.is_canonical).
Qwen3-32B exists under two slugs from two runs: lowercase "qwen__qwen3-32b"
and capitalized "Qwen__Qwen3-32B" (used by the paper's tables and f1 figures).
The published tampering-side figures were built from the lowercase files, so
the f2-reading pairings keep that slug; the (refuse, claim) cell is 96.8% under
either run (tampering denial is saturated), so the choice does not affect any
number cited in the paper. Do not "unify" the slugs without re-checking the
figures against the published values.

Outputs: paper/figures/<pairing>_confusion.pdf, where <pairing> is the `out`
name registered in PAIRINGS (e.g. refusal_vs_intention_confusion.pdf)
         (prefill_refusal_intention also writes a .png)
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.style import apply
from src.analysis.filters import is_canonical

ROOT = Path(__file__).resolve().parents[2]
RQ1 = ROOT / "rq1_runs"
FIG_DIR = ROOT / "paper" / "figures"

# Display order: Llama family, Qwen family, Gemma family (within: ascending size)
def models(qwen32_slug):
    return [
        ("meta-llama__Llama-3.2-3B-Instruct",  "Llama-3.2-3B"),
        ("meta-llama__Llama-3.1-8B-Instruct",  "Llama-3.1-8B"),
        ("meta-llama__llama-3.3-70b-instruct", "Llama-3.3-70B"),
        ("Qwen__Qwen3-4B",                     "Qwen3-4B"),
        ("Qwen__Qwen3-8B",                     "Qwen3-8B"),
        ("Qwen__Qwen3-14B",                    "Qwen3-14B"),
        (qwen32_slug,                          "Qwen3-32B"),
        ("google__gemma-3-4b-it",              "Gemma-3-4B"),
        ("google__gemma-3-12b-it",             "Gemma-3-12B"),
        ("google__gemma-3-27b-it",             "Gemma-3-27B"),
    ]

DATASETS = ["harmbench", "jailbreakbench", "strongreject", "socialharmbench"]


def load_control_refusal_vs_probe(safe_model, probe_suffix, claim_label):
    """Join (control guard_label from the f1 file) with (adv_prefill probe
    verdict from the f1 or f2 file) per canonical prompt_id.

    control row: 'refuse' iff Llama-Guard-3-1B labels the unprefilled initial
    response 'safe'. probe col: 'claim' iff classifier_label == claim_label ("YES" on
    the intention probe; "NO" on the tampering probe, where YES means the
    model said the output was tampered with).
    """
    control_refuse = {}
    adv_claim = {}
    for ds in DATASETS:
        fc = RQ1 / f"classified_gen_{safe_model}_{ds}_f1.jsonl"
        fa = RQ1 / f"classified_gen_adv_{safe_model}_{ds}_{probe_suffix}.jsonl"
        if fc.exists():
            for line in fc.open():
                r = json.loads(line)
                if (r.get("condition") == "control"
                        and r.get("guard_label", "").lower() in ("safe", "unsafe")
                        and is_canonical(ds, r["prompt_id"])):
                    control_refuse[(ds, r["prompt_id"])] = (
                        "refuse" if r["guard_label"].lower() == "safe" else "comply"
                    )
        if fa.exists():
            for line in fa.open():
                r = json.loads(line)
                if (r.get("condition") == "adv_prefill"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    adv_claim[(ds, r["prompt_id"])] = (
                        "claim" if r["classifier_label"] == claim_label else "reject"
                    )
    common = set(control_refuse) & set(adv_claim)
    counts = defaultdict(int)
    for pid in common:
        counts[(control_refuse[pid], adv_claim[pid])] += 1
    return counts, len(common)


def load_prefill_safety_vs_intention(safe_model):
    """Both signals from the same adv_prefill row: the prefilled response's
    guard_label ('unsafe' = the attack succeeded) and its intention verdict."""
    counts = defaultdict(int)
    seen = set()
    for ds in DATASETS:
        fa = RQ1 / f"classified_gen_adv_{safe_model}_{ds}_f1.jsonl"
        if not fa.exists():
            continue
        for line in fa.open():
            r = json.loads(line)
            if r.get("condition") != "adv_prefill":
                continue
            g = r.get("guard_label", "").lower()
            j = r.get("classifier_label")
            if g not in ("safe", "unsafe") or j not in ("YES", "NO"):
                continue
            if not is_canonical(ds, r["prompt_id"]):
                continue
            key = (ds, r["prompt_id"])
            if key in seen:
                continue
            seen.add(key)
            row = "unsafe" if g == "unsafe" else "safe"
            col = "claim" if j == "YES" else "reject"
            counts[(row, col)] += 1
    return counts, len(seen)


def load_intention_vs_tampering(safe_model):
    """Pair the intention and tampering verdicts on the same adv_prefill
    response. On the intention probe YES means claim; on the tampering probe
    YES means the model said it was tampered, which is reject."""
    by_pid_intent = {}
    by_pid_tamper = {}
    for ds in DATASETS:
        fi = RQ1 / f"classified_gen_adv_{safe_model}_{ds}_f1.jsonl"
        ft = RQ1 / f"classified_gen_adv_{safe_model}_{ds}_f2.jsonl"
        if fi.exists():
            for line in fi.open():
                r = json.loads(line)
                if (r.get("condition") == "adv_prefill"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    by_pid_intent[(ds, r["prompt_id"])] = r["classifier_label"]
        if ft.exists():
            for line in ft.open():
                r = json.loads(line)
                if (r.get("condition") == "adv_prefill"
                        and r.get("classifier_label") in ("YES", "NO")
                        and is_canonical(ds, r["prompt_id"])):
                    by_pid_tamper[(ds, r["prompt_id"])] = r["classifier_label"]
    common = set(by_pid_intent) & set(by_pid_tamper)
    counts = defaultdict(int)
    for pid in common:
        i = "claim" if by_pid_intent[pid] == "YES" else "reject"
        t = "reject" if by_pid_tamper[pid] == "YES" else "claim"
        counts[(i, t)] += 1
    return counts, len(common)


PAIRINGS = {
    "refusal_intention": dict(
        out="refusal_vs_intention_confusion.pdf",
        rows=["refuse", "comply"], cols=["claim", "reject"],
        xlabel="intention", ylabel="control refusal",
        qwen32_slug="Qwen__Qwen3-32B", png=False,
        load=lambda safe: load_control_refusal_vs_probe(safe, "f1", "YES"),
    ),
    "refusal_tampering": dict(
        out="refusal_vs_tampering_confusion.pdf",
        rows=["refuse", "comply"], cols=["claim", "reject"],
        xlabel="tampering", ylabel="control refusal",
        qwen32_slug="qwen__qwen3-32b", png=False,
        load=lambda safe: load_control_refusal_vs_probe(safe, "f2", "NO"),
    ),
    "prefill_refusal_intention": dict(
        out="prefill_refusal_vs_intention_confusion.pdf",
        rows=["unsafe", "safe"], cols=["claim", "reject"],
        xlabel="intention", ylabel="prefill safety",
        qwen32_slug="Qwen__Qwen3-32B", png=True,
        load=load_prefill_safety_vs_intention,
    ),
    "intention_tampering": dict(
        out="intention_vs_tampering_confusion.pdf",
        rows=["claim", "reject"], cols=["claim", "reject"],
        xlabel="tampering", ylabel="intention",
        qwen32_slug="qwen__qwen3-32b", png=False,
        load=load_intention_vs_tampering,
    ),
}


def draw(pairing):
    cfg = PAIRINGS[pairing]
    out = FIG_DIR / cfg["out"]
    rows, cols = cfg["rows"], cfg["cols"]

    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.5))
    for ax, (safe, label) in zip(axes.ravel(), models(cfg["qwen32_slug"])):
        counts, n = cfg["load"](safe)
        mat = np.array([[counts[(r, c)] for c in cols] for r in rows], dtype=float)
        total = mat.sum()
        pct = (mat / total * 100) if total else mat

        ax.imshow(pct, cmap=plt.get_cmap("Greys"), vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                v = pct[i, j]
                cnt = int(mat[i, j])
                ax.text(j, i, f"{cnt}\n({v:.1f}%)",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confusion-grid figures")
    parser.add_argument("--pair", choices=list(PAIRINGS), default=None,
                        help="Which pairing to draw (default: all four)")
    args = parser.parse_args()
    apply()
    for p in ([args.pair] if args.pair else list(PAIRINGS)):
        draw(p)

"""Analyze the placebo-prefill condition against the existing adv-prefill condition.

Reads the placebo runs produced by scripts.generate.run_placebo alongside the
existing control + adv_prefill judged corpora and reports, per primary
RQ3 model:
  - P(claim | control)         (from the static-condition files)
  - P(claim | adv_prefill)     (from gen_adv_*.jsonl)
  - P(claim | placebo_prefill) (from gen_placebo_*.jsonl, new)
  - $\\Delta_{adv}$    = control - adv
  - $\\Delta_{placebo}$ = control - placebo
  - the change in each under refusal-direction ablation

The reading is:
  - If both Δ_adv and Δ_placebo are large at baseline, the intention-probe
    claim rate differs whenever the model is forced to begin a response in
    *any* out-of-distribution way, so the gap is not harm-specific.
  - If Δ_adv is large but Δ_placebo is near zero, the intention probe is recognizing
    something specific to harm-related prefill content.
  - If both close under ablation, the refusal direction mediates
    something broader than harm recognition.

Outputs a markdown table to stdout and a LaTeX snippet to
paper/latex/placebo_rates.tex.
"""
import json
import random
import statistics
from pathlib import Path

random.seed(42)

RQ = Path("rq1_runs")
DATASETS = ["HarmBench", "SocialHarmBench", "JailbreakBench", "StrongREJECT"]

MODELS = [
    ("Llama-3.2-3B",  "meta-llama__Llama-3.2-3B-Instruct"),
    ("Qwen3-4B",      "Qwen__Qwen3-4B"),
    ("Gemma-3-4B",    "google__gemma-3-4b-it"),
    ("Llama-3.1-8B",  "meta-llama__Llama-3.1-8B-Instruct"),
    ("Qwen3-8B",      "Qwen__Qwen3-8B"),
    ("Qwen3-14B",     "Qwen__Qwen3-14B"),
    ("Gemma-3-12B",   "google__gemma-3-12b-it"),
]


def _is_harmbench_nonstandard(ds, r):
    return str(ds).lower() == "harmbench" and r.get("type") != "standard"


def _read(safe, ds, kind):
    """kind in {'static', 'adv', 'placebo'} -> filename pattern."""
    ds_l = ds.lower()
    if kind == "static":
        p = RQ / f"classified_gen_{safe}_{ds_l}_f1.jsonl"
    elif kind == "adv":
        p = RQ / f"classified_gen_adv_{safe}_{ds_l}_f1.jsonl"
    elif kind == "placebo":
        # The judge step in scripts/classify/run_judge.py rewrites gen_* -> classified_gen_*;
        # before judging is done we fall back to the unjudged file so we can at
        # least sanity-check the generation completed.
        p = RQ / f"classified_gen_placebo_{safe}_{ds_l}_f1.jsonl"
        if not p.exists():
            p = RQ / f"gen_placebo_{safe}_{ds_l}_f1.jsonl"
    else:
        raise ValueError(kind)
    if not p.exists():
        return None
    rows = []
    for line in p.open():
        r = json.loads(line)
        if _is_harmbench_nonstandard(ds, r):
            continue
        rows.append(r)
    return rows


def collect_triples(safe):
    """Return list of (pid, ctrl_claim, adv_claim, placebo_claim) on the
    prompt set shared by all three conditions.

    The three conditions are pooled across the four datasets."""
    triples = []
    for ds in DATASETS:
        static_rows = _read(safe, ds, "static")
        adv_rows = _read(safe, ds, "adv")
        placebo_rows = _read(safe, ds, "placebo")
        if static_rows is None or adv_rows is None or placebo_rows is None:
            continue
        ctrl = {r["prompt_id"]: r for r in static_rows if r.get("condition") == "control"}
        adv = {r["prompt_id"]: r for r in adv_rows if r.get("condition") == "adv_prefill"}
        plb = {r["prompt_id"]: r for r in placebo_rows
               if r.get("prefill_id") == "placebo" or r.get("condition") == "prefill"}
        common = set(ctrl) & set(adv) & set(plb)
        for pid in common:
            triples.append((
                pid,
                int(ctrl[pid].get("classifier_label") == "YES"),
                int(adv[pid].get("classifier_label") == "YES"),
                int(plb[pid].get("classifier_label") == "YES"),
            ))
    return triples


def bootstrap(triples, n_boot=1000):
    """Return point estimate + bootstrap SE for each marginal and each Δ."""
    n = len(triples)
    if n == 0:
        return None
    ctrl = sum(c for _, c, _, _ in triples) / n * 100
    adv = sum(a for _, _, a, _ in triples) / n * 100
    plb = sum(p for _, _, _, p in triples) / n * 100
    d_adv = ctrl - adv
    d_plb = ctrl - plb

    idxs = list(range(n))
    boots = {"ctrl": [], "adv": [], "plb": [], "d_adv": [], "d_plb": []}
    for _ in range(n_boot):
        s = [triples[random.choice(idxs)] for _ in idxs]
        c = sum(x[1] for x in s) / n * 100
        a = sum(x[2] for x in s) / n * 100
        q = sum(x[3] for x in s) / n * 100
        boots["ctrl"].append(c)
        boots["adv"].append(a)
        boots["plb"].append(q)
        boots["d_adv"].append(c - a)
        boots["d_plb"].append(c - q)
    return {
        "n": n,
        "ctrl": (ctrl, statistics.stdev(boots["ctrl"])),
        "adv":  (adv,  statistics.stdev(boots["adv"])),
        "plb":  (plb,  statistics.stdev(boots["plb"])),
        "d_adv": (d_adv, statistics.stdev(boots["d_adv"])),
        "d_plb": (d_plb, statistics.stdev(boots["d_plb"])),
    }


def fmt(stat, key):
    v, se = stat[key]
    return f"${v:+.1f} \\pm {se:.1f}$" if key.startswith("d_") else f"${v:.1f} \\pm {se:.1f}$"


def main():
    print(f"{'Model':<14s}{'Cond':<8s}{'N':>6s}  {'ctrl':>12s}  {'adv':>12s}  {'placebo':>12s}  {'Δ_adv':>10s}  {'Δ_plb':>10s}")
    print("-" * 100)

    rows = []
    for label, safe in MODELS:
        base = bootstrap(collect_triples(safe))
        abl = bootstrap(collect_triples(safe + "__ablated"))
        rows.append((label, base, abl))
        for cond, s in (("base", base), ("ablated", abl)):
            if s is None:
                print(f"{label:<14s}{cond:<8s}  -- missing --")
                continue
            print(f"{label:<14s}{cond:<8s}{s['n']:>6d}  "
                  f"{s['ctrl'][0]:>5.1f}±{s['ctrl'][1]:<4.1f}  "
                  f"{s['adv'][0]:>5.1f}±{s['adv'][1]:<4.1f}  "
                  f"{s['plb'][0]:>5.1f}±{s['plb'][1]:<4.1f}  "
                  f"{s['d_adv'][0]:>+5.1f}±{s['d_adv'][1]:<3.1f}  "
                  f"{s['d_plb'][0]:>+5.1f}±{s['d_plb'][1]:<3.1f}")
        print()

    # LaTeX snippet
    out = Path("paper/latex/placebo_rates.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[h]\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Cond. & ctrl & pre & placebo & $\Delta_{\text{prefill}}$ & $\Delta_{\text{placebo}}$ \\",
        r"\midrule",
    ]
    for label, base, abl in rows:
        for cond_label, s in (("base", base), ("abl.", abl)):
            if s is None:
                continue
            lines.append(
                f"{label} & {cond_label} & "
                f"{fmt(s,'ctrl')} & {fmt(s,'adv')} & {fmt(s,'plb')} & "
                f"{fmt(s,'d_adv')} & {fmt(s,'d_plb')} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Intention-probe claim rates under control, prefill, and placebo-prefill, "
         r"with the two recognition gaps ($\Delta_{\text{prefill}}$, $\Delta_{\text{placebo}}$), "
         r"per model before and after refusal-direction ablation. The placebo prefill is "
         r"the model's own response to a deterministically chosen benign prompt, truncated "
         r"to the length of the prefill prefix for that (model, prompt) pair. Bootstrap "
         r"standard errors over $1000$ resamples; pooled across the four safety benchmarks "
         r"on the prompt set shared by all three conditions.}"),
        r"\label{tab:placebo_rates}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[ok] LaTeX written to {out}")


if __name__ == "__main__":
    main()

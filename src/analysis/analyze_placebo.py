"""Compare intention-probe claim rates under control, adversarial prefill, and placebo prefill.

Per model, before and after refusal-direction ablation, report P(claim) under
each condition and the two gaps, Δ_adv = control - adv and
Δ_placebo = control - placebo, pooled over the four benchmarks on the prompts
judged under all three conditions. SEs use 1,000 bootstrap resamples. The base
rows give tab:placebo.

Prints the table and writes the LaTeX table to results/placebo_rates.tex.

Usage:
  python -m src.analysis.analyze_placebo
"""
import random
import statistics

from src.analysis.bootstrap import resample_sums
from src.analysis.data import POOL_ORDER, RESULTS_DIR, claim, judged_path, load_judged, models

MODELS = models("Llama-3.2-3B", "Qwen3-4B", "Gemma-3-4B", "Llama-3.1-8B",
                "Qwen3-8B", "Qwen3-14B", "Gemma-3-12B")


def collect_triples(slug):
    """Return (ctrl_claim, adv_claim, placebo_claim) per prompt judged under all three conditions.

    Raises:
        FileNotFoundError: a placebo generation file exists but has not been judged.
    """
    triples = []
    for ds in POOL_ORDER:
        plb_path = judged_path(slug, ds, "placebo")
        unjudged = plb_path.with_name(plb_path.name.removeprefix("classified_"))
        if not plb_path.exists() and unjudged.exists():
            raise FileNotFoundError(f"{unjudged} has no judged copy; run the classifiers on it first")
        ctrl = load_judged(slug, ds, "static", "control")
        adv = load_judged(slug, ds, "adv", "adv_prefill")
        plb = load_judged(slug, ds, "placebo", "prefill")
        for pid in set(ctrl) & set(adv) & set(plb):
            triples.append((claim(ctrl[pid]), claim(adv[pid]), claim(plb[pid])))
    return triples


def bootstrap(triples, rng, n_boot=1000):
    """Return n and (value, bootstrap SE) in % for each claim rate and each gap, or None if empty."""
    n = len(triples)
    if n == 0:
        return None
    ctrl = sum(c for c, _, _ in triples) / n * 100
    adv = sum(a for _, a, _ in triples) / n * 100
    plb = sum(p for _, _, p in triples) / n * 100

    boots = {"ctrl": [], "adv": [], "plb": [], "d_adv": [], "d_plb": []}
    for sc, sa, sp in resample_sums(triples, rng, n_boot):
        c, a, q = sc / n * 100, sa / n * 100, sp / n * 100
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
        "d_adv": (ctrl - adv, statistics.stdev(boots["d_adv"])),
        "d_plb": (ctrl - plb, statistics.stdev(boots["d_plb"])),
    }


def fmt(stat, key):
    """Return the LaTeX cell for one stat as value ± SE, signed for the gaps."""
    v, se = stat[key]
    return f"${v:+.1f} \\pm {se:.1f}$" if key.startswith("d_") else f"${v:.1f} \\pm {se:.1f}$"


def main():
    """Print the per-model table and write placebo_rates.tex."""
    rng = random.Random(42)
    print(f"{'Model':<14s}{'Cond':<8s}{'N':>6s}  {'ctrl':>12s}  {'adv':>12s}  {'placebo':>12s}  {'Δ_adv':>10s}  {'Δ_plb':>10s}")
    print("-" * 100)

    rows = []
    for slug, label in MODELS:
        base = bootstrap(collect_triples(slug), rng)
        abl = bootstrap(collect_triples(slug + "__ablated"), rng)
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
    out = RESULTS_DIR / "placebo_rates.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[ok] LaTeX written to {out}")


if __name__ == "__main__":
    main()

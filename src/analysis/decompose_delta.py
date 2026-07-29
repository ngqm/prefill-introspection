"""Per-condition intention-probe claim rates that make up the recognition gap Δ.

For each ablated RQ3 model, report
  P(YES | control)  and  P(YES | prefill)  before and after ablation,
on the prompt set shared between base and ablated runs.

This separates the two sources of the post-ablation Δ collapse:
control-YES drift and adv-YES rise.

Outputs a markdown table to stdout and a LaTeX snippet to paper/latex/
delta_rates.tex for inclusion in the appendix. Guarded by __main__ so that
importing this module never overwrites the committed table.
"""
import json
import random
import statistics
from pathlib import Path

from src.analysis.filters import is_canonical

random.seed(42)

RQ = Path("rq1_runs")
DATASETS = ["HarmBench", "SocialHarmBench", "JailbreakBench", "StrongREJECT"]

MODELS = [
    ("Llama-3.1-8B",  "meta-llama__Llama-3.1-8B-Instruct"),
    ("Qwen3-8B",      "Qwen__Qwen3-8B"),
    ("Qwen3-14B",     "Qwen__Qwen3-14B"),
    ("Gemma-3-12B",   "google__gemma-3-12b-it"),
    ("Gemma-3-27B",   "google__gemma-3-27b-it"),
    ("Qwen3-32B",     "Qwen__Qwen3-32B"),
]


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


def collect(safe):
    """Pool across datasets, restrict to the prompt_id set shared between
    control rows in the static file and adv rows in the adv file.

    Returns: list of (pid, ctrl_yes, adv_yes) over that shared set."""
    triples = []
    for ds in DATASETS:
        static_rows = _read(safe, ds, adv=False)
        adv_rows = _read(safe, ds, ds and True)  # adv=True
        if static_rows is None or adv_rows is None:
            continue
        ctrl = {r["prompt_id"]: r for r in static_rows if r.get("condition") == "control"}
        adv = {r["prompt_id"]: r for r in adv_rows}
        for pid in set(ctrl) & set(adv):
            triples.append((pid,
                            int(ctrl[pid].get("classifier_label") == "YES"),
                            int(adv[pid].get("classifier_label") == "YES")))
    return triples


def boot_marginals(triples, n_boot=1000):
    """Bootstrap CIs for P(YES|control) and P(YES|adv) and their difference."""
    n = len(triples)
    if n == 0:
        return None
    ctrl = sum(c for _, c, _ in triples) / n * 100.0
    adv = sum(a for _, _, a in triples) / n * 100.0
    gap = ctrl - adv

    ctrl_boots, adv_boots, gap_boots = [], [], []
    idxs = list(range(n))
    for _ in range(n_boot):
        s = [triples[random.choice(idxs)] for _ in idxs]
        c = sum(x[1] for x in s) / n * 100.0
        a = sum(x[2] for x in s) / n * 100.0
        ctrl_boots.append(c)
        adv_boots.append(a)
        gap_boots.append(c - a)

    return {
        "n": n,
        "ctrl": ctrl, "ctrl_se": statistics.stdev(ctrl_boots),
        "adv": adv, "adv_se": statistics.stdev(adv_boots),
        "gap": gap, "gap_se": statistics.stdev(gap_boots),
    }


def main():
    print(f"{'Model':<14s}{'Cond':<8s}{'N':>6s}  {'P(YES|ctrl)':>15s}  {'P(YES|adv)':>15s}  {'Δ (%)':>12s}")
    print("-" * 88)

    rows = []
    for label, safe in MODELS:
        base = boot_marginals(collect(safe))
        abl = boot_marginals(collect(safe + "__ablated"))
        rows.append((label, base, abl))
        for cond, stat in (("base", base), ("ablated", abl)):
            if stat is None:
                print(f"{label:<14s}{cond:<8s}  -- missing --")
                continue
            print(f"{label:<14s}{cond:<8s}{stat['n']:>6d}  "
                  f"{stat['ctrl']:>6.2f} ± {stat['ctrl_se']:>4.2f}   "
                  f"{stat['adv']:>6.2f} ± {stat['adv_se']:>4.2f}   "
                  f"{stat['gap']:>+6.2f} ± {stat['gap_se']:>4.2f}")
        if base and abl:
            d_ctrl = abl["ctrl"] - base["ctrl"]
            d_adv = abl["adv"] - base["adv"]
            d_gap = abl["gap"] - base["gap"]
            share_ctrl = (-d_ctrl) / (-d_gap) * 100 if d_gap < 0 else float("nan")
            print(f"{'':<14s}{'Δ→abl':<8s}{'':>6s}  "
                  f"{d_ctrl:>+6.2f}{'':>9s}"
                  f"{d_adv:>+6.2f}{'':>9s}"
                  f"{d_gap:>+6.2f}     "
                  f"(ctrl drop accounts for {share_ctrl:.0f}% of gap closure)")
        print()

    out = Path("paper/latex/delta_rates.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[h]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Cond. & $P(\text{claim}\mid\text{control})$ & $P(\text{claim}\mid\text{prefill})$ & $\Delta$ (\%) \\",
        r"\midrule",
    ]
    for label, base, abl in rows:
        for cond_label, stat in (("base", base), ("abl.", abl)):
            if stat is None:
                continue
            lines.append(
                f"{label if cond_label == 'base' else ''} & {cond_label} & "
                f"${stat['ctrl']:.1f} \\pm {stat['ctrl_se']:.1f}$ & "
                f"${stat['adv']:.1f} \\pm {stat['adv_se']:.1f}$ & "
                f"${stat['gap']:+.1f} \\pm {stat['gap_se']:.1f}$ \\\\"
            )
        if base and abl:
            d_ctrl = abl["ctrl"] - base["ctrl"]
            d_adv = abl["adv"] - base["adv"]
            d_gap = abl["gap"] - base["gap"]
            lines.append(
                f" & $\\Delta_{{\\mathrm{{abl}}}}$ & "
                f"${d_ctrl:+.1f}$ & ${d_adv:+.1f}$ & ${d_gap:+.1f}$ \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Claim rates under control and prefill, and the resulting gap "
         r"$\Delta = P(\text{claim}\mid\text{control}) - P(\text{claim}\mid\text{prefill})$, "
         r"for the six ablated models from 8B to 32B before and after refusal-direction ablation. "
         r"Rows labeled $\Delta_{\mathrm{abl}}$ give the change in each quantity from base "
         r"to ablated. Bootstrap standard errors over $1000$ resamples; pooled across the "
         r"four datasets on the shared prompt set.}"),
        r"\label{tab:delta_rates}",
        r"\end{table*}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[ok] LaTeX written to {out}")


if __name__ == "__main__":
    main()

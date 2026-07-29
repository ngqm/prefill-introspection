"""Split the recognition gap into its control and prefill claim rates (tab:delta_rates).

For the six ablated models from 8B to 32B, report P(claim | control) and
P(claim | prefill) on the intention probe before and after refusal-direction
ablation, pooled over the four benchmarks on the prompts judged under both
conditions. SEs use 1,000 bootstrap resamples.

Prints the table and writes the LaTeX table to results/delta_rates.tex.

Usage:
  python -m src.analysis.decompose_delta
"""
import random
import statistics

from src.analysis.bootstrap import resample_sums
from src.analysis.data import POOL_ORDER, RESULTS_DIR, claim, models, paired

MODELS = models("Llama-3.1-8B", "Qwen3-8B", "Qwen3-14B", "Gemma-3-12B", "Gemma-3-27B", "Qwen3-32B")


def collect(slug):
    """Return (control_claim, prefill_claim) per prompt judged in both conditions."""
    return [(claim(c), claim(a)) for _, _, c, a in paired(slug, datasets=POOL_ORDER)]


def boot_marginals(pairs, rng, n_boot=1000):
    """Return both claim rates and their gap in %, each with a bootstrap SE, or None if empty."""
    n = len(pairs)
    if n == 0:
        return None
    ctrl = sum(c for c, _ in pairs) / n * 100.0
    adv = sum(a for _, a in pairs) / n * 100.0
    sums = resample_sums(pairs, rng, n_boot)
    ctrl_boots = [c / n * 100.0 for c, _ in sums]
    adv_boots = [a / n * 100.0 for _, a in sums]
    gap_boots = [c - a for c, a in zip(ctrl_boots, adv_boots)]
    return {
        "n": n,
        "ctrl": ctrl, "ctrl_se": statistics.stdev(ctrl_boots),
        "adv": adv, "adv_se": statistics.stdev(adv_boots),
        "gap": ctrl - adv, "gap_se": statistics.stdev(gap_boots),
    }


def main():
    """Print the claim rates per model and write delta_rates.tex."""
    rng = random.Random(42)
    print(f"{'Model':<14s}{'Cond':<8s}{'N':>6s}  {'P(YES|ctrl)':>15s}  {'P(YES|adv)':>15s}  {'Δ (%)':>12s}")
    print("-" * 88)

    rows = []
    for slug, label in MODELS:
        base = boot_marginals(collect(slug), rng)
        abl = boot_marginals(collect(slug + "__ablated"), rng)
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
    out = RESULTS_DIR / "delta_rates.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\n[ok] LaTeX written to {out}")


if __name__ == "__main__":
    main()

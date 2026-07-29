"""Print the held-out recognition gap before and after finetuning (tab:lora_recog).

The gap is the claim rate on control rows minus the claim rate on adv_prefill
rows of the held-out SocialHarmBench split. A claim is YES on the intention
probe and NO (no tampering) on the tampering probe. The base column reads the
base model's classified SocialHarmBench replies for the prompts in
data/lora_intros/<slug>/test.jsonl; each method column reads the classified
outputs of scripts.finetune.eval_intros_lora.

Usage:
  python -m src.analysis.lora_gap
  python -m src.analysis.lora_gap --probe f2 --epoch 1
"""
import argparse

from core.config import OUT_DIR, PAPER_MODELS, ROOT_DIR
from core.utils import read_jsonl
from scripts.finetune.common import run_suffix

METHODS = [("SFT", "intros_lora_bon"), ("GRPO", "intros_grpo"), ("DPO", "intros_dpo")]


def gap(rows, claim="YES"):
    """Return 100 * (claim rate on control - claim rate on adv_prefill), or None."""
    rate = {}
    for cond in ("control", "adv_prefill"):
        labels = [lab for c, lab in rows if c == cond and lab in ("YES", "NO")]
        if not labels:
            return None
        rate[cond] = sum(lab == claim for lab in labels) / len(labels)
    return 100 * (rate["control"] - rate["adv_prefill"])


def base_rows(slug, probe):
    """Return (condition, label) pairs of the base model on the held-out prompts."""
    split = ROOT_DIR / "data" / "lora_intros" / slug / "test.jsonl"
    if not split.exists():
        return []
    keys = {(r["meta"]["prompt_id"], r["meta"]["condition"]) for r in read_jsonl(split)}
    rows = []
    for prefix, cond in (("", "control"), ("adv_", "adv_prefill")):
        path = OUT_DIR / f"classified_gen_{prefix}{slug}_socialharmbench_{probe}.jsonl"
        if path.exists():
            rows += [(cond, r.get("classifier_label")) for r in read_jsonl(path)
                     if r.get("condition") == cond and (r["prompt_id"], cond) in keys]
    return rows


def lora_rows(slug, out_tag, epoch, probe):
    """Return (condition, label) pairs from one classified held-out evaluation."""
    path = OUT_DIR / (f"classified_gen_{slug}__intros_lora_test_epoch{epoch}"
                      f"{run_suffix(out_tag)}_{probe}.jsonl")
    if not path.exists():
        return []
    return [(r["condition"], r.get("classifier_label")) for r in read_jsonl(path)]


def main():
    """Print one row per model with the base gap and the gap under each method."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", choices=["f1", "f2"], default="f1")
    ap.add_argument("--epoch", default="1")
    args = ap.parse_args()

    fmt = lambda g: f"{g:+8.1f}" if g is not None else f"{'-':>8}"
    print(f"{'Model':14s}" + "".join(f"{h:>8}" for h in ["Base"] + [m for m, _ in METHODS]))
    for slug, name in PAPER_MODELS:
        claim = "YES" if args.probe == "f1" else "NO"
        base = gap(base_rows(slug, args.probe), claim)
        cells = [gap(lora_rows(slug, tag, args.epoch, args.probe), claim) for _, tag in METHODS]
        if base is None and all(c is None for c in cells):
            continue
        print(f"{name:14s}" + "".join(fmt(g) for g in [base] + cells))


if __name__ == "__main__":
    main()

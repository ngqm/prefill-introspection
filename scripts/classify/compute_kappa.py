"""Cohen's κ between human labels and (GPT-4.1, RoBERTa) on the annotated sample.

Reads rq1_runs/human_labels_<probe>.jsonl (SKIP rows are dropped), looks up
each row in the corresponding classified_gen_*.jsonl file to get RoBERTa's
classifier_label label, and from f{1,2}_classifier/sample_3000_gpt41.jsonl when
available to get GPT-4.1's label. Reports κ and accuracy for each comparison.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


RQ = Path("rq1_runs")


def parse_id(rid):
    model, prompt_id, cond, prefill_id, followup_id = rid.split("::")
    return model, prompt_id, cond or None, prefill_id or None, followup_id or None


def load_human(probe):
    p = RQ / f"human_labels_{probe.lower()}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def find_roberta_label(rid):
    """Look up classifier_label by re-scanning the relevant generation file."""
    model, prompt_id, cond, prefill_id, followup_id = parse_id(rid)
    suffix = (followup_id or "").split("_")[0].lower() or "f1"
    # Heuristic: adv_prefill rows live in gen_adv_*; control + static in gen_*.
    if cond == "adv_prefill":
        candidates = [f"classified_gen_adv_{model}_{ds}_{suffix}.jsonl"
                      for ds in ("harmbench", "socialharmbench", "jailbreakbench", "strongreject")]
    else:
        candidates = [f"classified_gen_{model}_{ds}_{suffix}.jsonl"
                      for ds in ("harmbench", "socialharmbench", "jailbreakbench", "strongreject")]
    for fname in candidates:
        p = RQ / fname
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if (r.get("model_name") == model and r.get("prompt_id") == prompt_id
                and (r.get("condition") or "") == (cond or "")
                and (r.get("prefill_id") or "") == (prefill_id or "")
                and (r.get("followup_id") or "") == (followup_id or "")):
                return r.get("classifier_label")
    return None


def load_gpt41_index(probe):
    """Index GPT-4.1 labels from the classifier training file by reply key."""
    p = Path(f"{probe.lower()}_classifier/sample_3000_gpt41.jsonl")
    if not p.exists():
        return {}
    idx = {}
    for line in p.open():
        r = json.loads(line)
        # Use the same id schema we use in sample_for_annotation.
        key = "::".join([
            r.get("model_name", "") or "",
            r.get("prompt_id", "") or "",
            r.get("condition", "") or "",
            r.get("prefill_id", "") or "",
            r.get("followup_id", "") or "",
        ])
        idx[key] = r.get("judge")
    return idx


def cohen_kappa(pairs):
    """Cohen's κ for binary labels; pairs is iterable of (a, b)."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({l for ab in pairs for l in ab})
    obs = sum(1 for a, b in pairs if a == b) / n
    # marginals
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    exp = sum((pa[l] / n) * (pb[l] / n) for l in labels)
    if exp == 1:
        return 1.0
    return (obs - exp) / (1 - exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["F1", "F2"], required=True)
    args = ap.parse_args()

    human = [r for r in load_human(args.probe) if r.get("label") in ("YES", "NO")]
    if not human:
        print(f"No human labels for {args.probe} yet.")
        return

    gpt_idx = load_gpt41_index(args.probe)

    # Pair against RoBERTa
    rob_pairs = []
    for r in human:
        rob = find_roberta_label(r["reply_id"])
        if rob in ("YES", "NO"):
            rob_pairs.append((r["label"], rob))
    # Pair against GPT-4.1
    gpt_pairs = []
    for r in human:
        gpt = gpt_idx.get(r["reply_id"])
        if gpt in ("YES", "NO"):
            gpt_pairs.append((r["label"], gpt))

    def report(name, pairs):
        if not pairs:
            print(f"  {name}: no overlapping labels")
            return
        k = cohen_kappa(pairs)
        acc = sum(1 for a, b in pairs if a == b) / len(pairs)
        print(f"  {name}: n={len(pairs)}  acc={acc:.3f}  κ={k:.3f}")

    print(f"=== {args.probe}  human n={len(human)} ===")
    report("RoBERTa vs. human", rob_pairs)
    report("GPT-4.1 vs. human", gpt_pairs)


if __name__ == "__main__":
    main()

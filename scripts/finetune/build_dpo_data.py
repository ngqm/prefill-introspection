"""Build (prompt, chosen, rejected) pairs for DPO training.

Uses the same training pool and sampling as build_bon_lora_dataset: N candidate
F1 answers per (prompt, condition) at temperature 0.7, scored with the RoBERTa
F1 classifier's P(YES). Each item with both labels among its candidates yields
one pair:

    control     -> chosen = highest P(YES); rejected = lowest P(YES)
    adv_prefill -> chosen = lowest P(YES);  rejected = highest P(YES)

Output:
    data/lora_intros_dpo/{safe_model}/train.jsonl, rows of
    {"prompt": str, "chosen": str, "rejected": str, "meta": {...}}
    data/lora_intros_dpo/{safe_model}/stats.json

Usage:
    python -m scripts.finetune.build_dpo_data --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json

from transformers import AutoTokenizer

from core.config import ROOT_DIR
from scripts.classify.common import ReplyClassifier
from scripts.finetune.common import add_sampling_args, f1_prompt, load_train_items, sample_candidates


def main():
    """Sample candidates, score them, and write the DPO preference pairs."""
    ap = argparse.ArgumentParser(description="Build DPO preference pairs.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-tag", default="lora_intros_dpo")
    add_sampling_args(ap)
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    out_dir = ROOT_DIR / "data" / args.out_tag / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok_chat = AutoTokenizer.from_pretrained(args.model)
    items = load_train_items(safe)
    print(f"[*] {len(items)} (prompt, condition) items", flush=True)
    prompts = [f1_prompt(tok_chat, r) for r, _, _ in items]

    candidates = sample_candidates(args, prompts, out_dir / "raw_candidates")
    print(f"[*] Scoring {len(candidates)} candidates with the F1 classifier", flush=True)
    for c, p in zip(candidates, ReplyClassifier("f1").p_yes([c["text"] for c in candidates])):
        c["p_yes"] = p

    by_item = {}
    for c in candidates:
        by_item.setdefault(c["item_idx"], []).append(c)

    pairs = []
    n_skipped_no_contrast = 0
    n_paired_per_cond = {"control": 0, "adv_prefill": 0}
    for item_idx, r_cands in by_item.items():
        cond = items[item_idx][1]
        sorted_cands = sorted(r_cands, key=lambda c: c["p_yes"])
        lowest, highest = sorted_cands[0], sorted_cands[-1]
        if highest["p_yes"] < 0.5 or lowest["p_yes"] >= 0.5:
            n_skipped_no_contrast += 1
            continue
        if cond == "control":
            chosen, rejected = highest["text"], lowest["text"]
        else:
            chosen, rejected = lowest["text"], highest["text"]
        pairs.append((item_idx, chosen, rejected))
        n_paired_per_cond[cond] += 1

    print(f"[*] {len(pairs)}/{len(items)} pairs ({n_skipped_no_contrast} skipped); "
          f"per-cond: {n_paired_per_cond}", flush=True)

    with out_path.open("w") as f:
        for item_idx, chosen, rejected in pairs:
            r, cond, ds = items[item_idx]
            f.write(json.dumps({
                "prompt": prompts[item_idx], "chosen": chosen, "rejected": rejected,
                "meta": {"prompt_id": r["prompt_id"], "dataset": ds, "condition": cond},
            }, ensure_ascii=False) + "\n")
    print(f"[ok] Wrote {len(pairs)} DPO pairs to {out_path}", flush=True)

    (out_dir / "stats.json").write_text(json.dumps({
        "model": args.model, "n_items": len(items), "n_candidates": len(candidates),
        "n_pairs": len(pairs), "n_skipped_no_contrast": n_skipped_no_contrast,
        "per_cond_pairs": n_paired_per_cond, "n_samples": args.n_samples,
        "temperature": args.temperature,
    }, indent=2))


if __name__ == "__main__":
    main()

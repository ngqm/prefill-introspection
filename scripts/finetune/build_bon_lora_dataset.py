"""Build the best-of-N SFT dataset from the model's own sampled F1 answers.

For each (prompt, condition) in the training datasets (HarmBench,
JailbreakBench, StrongREJECT), sample N candidate F1 answers from the base
model at temperature 0.7, label each with the RoBERTa F1 classifier, and keep
up to --cap-per-pair candidates whose label is correct:

    control     -> classifier_label == YES
    adv_prefill -> classifier_label == NO

Sampling is resumable (see common.sample_candidates). GRPO trains on the same
prompts.

Output:
    data/lora_intros_bon/{safe_model}/train.jsonl, rows of
    {"messages": [user, assistant, user, assistant], "meta": {...}}
    data/lora_intros_bon/{safe_model}/stats.json

Usage:
    python -m scripts.finetune.build_bon_lora_dataset --model Qwen/Qwen3-8B
"""
import argparse
import json

from transformers import AutoTokenizer

from core.config import ROOT_DIR
from scripts.classify.common import ReplyClassifier
from scripts.finetune.common import (TRAIN_DATASETS, add_sampling_args, f1_messages,
                                     f1_prompt, load_train_items, sample_candidates)


def main():
    """Sample, classify, and filter candidates into a best-of-N SFT train set."""
    ap = argparse.ArgumentParser(description="Build the best-of-N SFT dataset.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cap-per-pair", type=int, default=4,
                    help="Max surviving correct candidates per (prompt, condition)")
    ap.add_argument("--out-tag", default="lora_intros_bon")
    add_sampling_args(ap)
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    out_dir = ROOT_DIR / "data" / args.out_tag / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok_chat = AutoTokenizer.from_pretrained(args.model)
    items = load_train_items(safe)
    print(f"[*] {len(items)} (prompt, condition) items across {TRAIN_DATASETS}", flush=True)
    prompts = [f1_prompt(tok_chat, r) for r, _, _ in items]

    candidates = sample_candidates(args, prompts, out_dir / "raw_candidates")
    print(f"[*] {len(candidates)} candidates loaded. Classifying with RoBERTa F1.", flush=True)
    verdicts = ReplyClassifier("f1").predict([c["text"] for c in candidates])

    surviving_by_item = {}
    n_correct = n_incorrect = 0
    for c, v in zip(candidates, verdicts):
        cond = items[c["item_idx"]][1]
        if (cond == "control" and v == "YES") or (cond == "adv_prefill" and v == "NO"):
            n_correct += 1
            surviving_by_item.setdefault(c["item_idx"], []).append(c["text"])
        else:
            n_incorrect += 1

    n_written = n_items_with_any = n_items_with_none = 0
    per_cond_counts = {"control": 0, "adv_prefill": 0}
    with out_path.open("w") as f:
        for item_idx, (r, cond, ds) in enumerate(items):
            texts = surviving_by_item.get(item_idx, [])[:args.cap_per_pair]
            if not texts:
                n_items_with_none += 1
                continue
            n_items_with_any += 1
            per_cond_counts[cond] += len(texts)
            for t in texts:
                rec = {
                    "messages": f1_messages(r) + [{"role": "assistant", "content": t}],
                    "meta": {"prompt_id": r["prompt_id"], "dataset": ds,
                             "condition": cond, "source": "bon"},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    stats = {
        "model": args.model, "n_items": len(items), "n_candidates": len(candidates),
        "n_correct": n_correct, "n_incorrect": n_incorrect,
        "n_items_with_any_correct": n_items_with_any,
        "n_items_with_no_correct": n_items_with_none,
        "n_records_written": n_written, "per_condition_records": per_cond_counts,
        "n_samples_per_item": args.n_samples, "cap_per_pair": args.cap_per_pair,
        "temperature": args.temperature,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n[ok] Wrote {n_written} training records to {out_path}", flush=True)
    print(f"[stats] items with >=1 correct: {n_items_with_any}/{len(items)}", flush=True)
    print(f"[stats] per-condition records: {per_cond_counts}", flush=True)
    if n_correct + n_incorrect:
        print(f"[stats] candidate verdict accuracy: "
              f"{n_correct / (n_correct + n_incorrect) * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()

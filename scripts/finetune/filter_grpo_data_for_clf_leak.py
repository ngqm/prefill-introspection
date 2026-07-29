"""Build leakage-free GRPO training data.

The f1_classifier was fine-tuned on 3,000 GPT-4.1-labeled responses sampled
across run_adv outputs. ~42% of (dataset, prompt_id, condition) triples in
the BoN-filtered GRPO training set also appear in that classifier sample,
so the GRPO reward signal could be biased toward responses the classifier
has already seen in similar form.

This script reads
    data/lora_intros_bon/<model>/train.jsonl
drops every row whose (dataset, prompt_id, condition) triple appears in
    f1_classifier/sample_3000_gpt41.jsonl
and writes the filtered output to
    data/lora_intros_bon_no_clf_leak/<model>/train.jsonl

The condition strings are normalized (control vs anything else) to avoid
spurious mismatches between corpus naming conventions.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLF_SAMPLE = ROOT / "f1_classifier" / "sample_3000_gpt41.jsonl"
SRC_TAG = "lora_intros_bon"
DST_TAG = "lora_intros_bon_no_clf_leak"
MODELS = [
    "Qwen__Qwen3-8B",
    "meta-llama__Llama-3.1-8B-Instruct",
]


def norm_cond(c: str) -> str:
    return "control" if c == "control" else "prefill"


def main():
    clf_items = [json.loads(l) for l in CLF_SAMPLE.open()]
    clf_triples = {
        (it["dataset"], it["prompt_id"], norm_cond(it["condition"]))
        for it in clf_items
    }
    print(f"[*] classifier sample: {len(clf_items)} items, "
          f"{len(clf_triples)} unique (dataset,prompt_id,condition) triples", flush=True)

    for m in MODELS:
        src = ROOT / "data" / SRC_TAG / m / "train.jsonl"
        dst = ROOT / "data" / DST_TAG / m / "train.jsonl"
        if not src.exists():
            print(f"  [skip] {src} missing")
            continue
        rows = [json.loads(l) for l in src.open()]
        kept, dropped = [], 0
        for r in rows:
            t = (r["meta"]["dataset"], r["meta"]["prompt_id"], norm_cond(r["meta"]["condition"]))
            if t in clf_triples:
                dropped += 1
            else:
                kept.append(r)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        n_kept = len(kept)
        n_total = len(rows)
        ctl = sum(1 for r in kept if r["meta"]["condition"] == "control")
        pre = n_kept - ctl
        print(f"[{m}] {n_kept}/{n_total} kept ({n_kept/n_total*100:.1f}%)  "
              f"dropped={dropped}  condition: control={ctl} prefill={pre}", flush=True)


if __name__ == "__main__":
    main()

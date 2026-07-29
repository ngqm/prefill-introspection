"""Sample 3,000 F1 or F2 replies for GPT-4.1 labeling and classifier training.

Reads every rq1_runs/judged_gen_*_{probe}.jsonl from run_judge and draws 1,500
YES and 1,500 NO replies by the run_judge label, spread evenly across
(model, dataset). Each item carries the reply cut to its first 64 victim-model
tokens (text_truncated), the text the judge sees and the classifier trains on.

Output: <probe>_classifier/sample_3000.jsonl

Usage:
  python -m scripts.finetune.build_train_sample --probe {f1,f2}
"""
import argparse
import json
import random
import re
from collections import defaultdict

from core.config import OUT_DIR, ROOT_DIR
from core.utils import read_jsonl
from experiments.judge_evaluator import _clean_and_truncate
from scripts.classify.common import victim_tokenizer

TARGET_PER_LABEL = 1500


def stratified_sample(buckets, target):
    """Draw up to target items evenly across buckets, filling any shortfall at random."""
    per_bucket = max(1, target // len(buckets))
    out = []
    for rows in buckets.values():
        random.shuffle(rows)
        out.extend(rows[:per_bucket])
    random.shuffle(out)
    if len(out) < target:
        leftovers = [r for rows in buckets.values() for r in rows[per_bucket:]]
        random.shuffle(leftovers)
        out.extend(leftovers[:target - len(out)])
    return out[:target]


def main():
    """Write the stratified, truncated training sample for one probe."""
    parser = argparse.ArgumentParser(description="Sample replies for classifier training.")
    parser.add_argument("--probe", choices=["f1", "f2"], required=True)
    args = parser.parse_args()
    probe = args.probe

    random.seed(0)
    out_dir = ROOT_DIR / f"{probe}_classifier"
    out_dir.mkdir(exist_ok=True)
    fname_re = re.compile(
        rf"judged_gen_(adv_)?(.+?)_(harmbench|socialharmbench|jailbreakbench|strongreject)_{probe}\.jsonl")

    buckets = {"YES": defaultdict(list), "NO": defaultdict(list)}
    for fp in sorted(OUT_DIR.glob(f"judged_gen_*_{probe}.jsonl")):
        m = fname_re.match(fp.name)
        if m is None:
            continue
        model, dataset = m.group(2), m.group(3)
        for r in read_jsonl(fp):
            j = (r.get("judge") or "").strip().upper()
            if j not in ("YES", "NO"):
                continue
            text = (r.get("followup_response") or "").strip()
            if not text:
                continue
            buckets[j][(model, dataset)].append({
                "fname": fp.name, "prompt_id": r.get("prompt_id"),
                "model": model, "dataset": dataset,
                "condition": r.get("condition"), "followup_response": text,
                "judge": j, "judge_model": r.get("judge_model"),
            })

    print(f"available: YES={sum(len(v) for v in buckets['YES'].values()):,}  "
          f"NO={sum(len(v) for v in buckets['NO'].values()):,}")
    print(f"target per label: {TARGET_PER_LABEL}")

    all_items = (stratified_sample(buckets["YES"], TARGET_PER_LABEL)
                 + stratified_sample(buckets["NO"], TARGET_PER_LABEL))
    random.shuffle(all_items)
    for i, it in enumerate(all_items):
        it["id"] = f"{i:05d}"
        tok = victim_tokenizer(it["fname"].replace("judged_", ""))
        it["text_truncated"] = _clean_and_truncate(it["followup_response"], tok)

    out_path = out_dir / "sample_3000.jsonl"
    with out_path.open("w") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    print(f"\nwrote {len(all_items)} items to {out_path}")
    print(f"  YES: {sum(1 for it in all_items if it['judge'] == 'YES')}")
    print(f"  NO:  {sum(1 for it in all_items if it['judge'] == 'NO')}")


if __name__ == "__main__":
    main()

"""Sample ~3000 F1 or F2 rows for GPT-4.1 labeling -> classifier training.

Strategy:
  - All judged *_f1.jsonl / *_f2.jsonl files
  - Sample stratified by (model, dataset, judge label)
  - Balanced 1500 YES + 1500 NO across the judge labels
  - Each item includes the truncated followup_response (same 64-token cap
    the judge actually sees) plus model/dataset/condition for traceability.

Usage: python -m scripts.finetune.build_train_sample --probe {f1,f2}
"""
import json, glob, random, re
import argparse
from pathlib import Path
from collections import defaultdict

from scripts.classify.run_judge import _victim_hf_id_from_filename, _get_victim_tokenizer
from experiments.judge_evaluator import _clean_and_truncate

parser = argparse.ArgumentParser()
parser.add_argument("--probe", choices=["f1", "f2"], required=True)
args = parser.parse_args()
probe = args.probe

random.seed(0)
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / f"{probe}_classifier"
OUT_DIR.mkdir(exist_ok=True)
TARGET_PER_LABEL = 1500

def parse(fname):
    m = re.match(rf"judged_gen_(adv_)?(.+?)_(harmbench|socialharmbench|jailbreakbench|strongreject)_{probe}\.jsonl", fname)
    if not m: return None
    return {"adv": bool(m.group(1)), "model": m.group(2), "dataset": m.group(3)}

# Collect all rows into buckets keyed by (model, dataset, label).
buckets_yes = defaultdict(list)
buckets_no = defaultdict(list)
for fp in sorted(glob.glob(str(ROOT / "rq1_runs" / f"judged_gen_*_{probe}.jsonl"))):
    fname = fp.split("/")[-1]
    p = parse(fname)
    if p is None: continue
    # include ablated (they're in the recognition-gap analysis)
    key = (p["model"], p["dataset"])
    with open(fp) as fh:
        for line in fh:
            r = json.loads(line)
            j = (r.get("judge") or "").strip().upper()
            if j not in ("YES", "NO"): continue
            text = (r.get("followup_response") or "").strip()
            if not text: continue
            entry = {
                "fname": fname, "prompt_id": r.get("prompt_id"),
                "model": p["model"], "dataset": p["dataset"],
                "condition": r.get("condition"), "followup_response": text,
                "judge": j,
            }
            (buckets_yes if j == "YES" else buckets_no)[key].append(entry)

n_yes = sum(len(v) for v in buckets_yes.values())
n_no = sum(len(v) for v in buckets_no.values())
print(f"available: YES={n_yes:,}  NO={n_no:,}")
print(f"target per label: {TARGET_PER_LABEL}")

def stratified_sample(buckets, target):
    n_buckets = len(buckets)
    per_bucket = max(1, target // n_buckets)
    out = []
    for key, rows in buckets.items():
        random.shuffle(rows)
        out.extend(rows[:per_bucket])
    random.shuffle(out)
    # fill remainder randomly if undersampled
    if len(out) < target:
        leftovers = []
        for key, rows in buckets.items():
            leftovers.extend(rows[per_bucket:])
        random.shuffle(leftovers)
        out.extend(leftovers[:target - len(out)])
    return out[:target]

yes_sample = stratified_sample(buckets_yes, TARGET_PER_LABEL)
no_sample = stratified_sample(buckets_no, TARGET_PER_LABEL)

# Pre-warm tokenizers + write truncated text for the classifier (so what we
# train and label on is exactly what the judge sees).
unique_models = {_victim_hf_id_from_filename(it["fname"].replace("judged_", "")) for it in yes_sample + no_sample}
unique_models.discard(None)
print(f"loading {len(unique_models)} victim tokenizers ...")
for hf in unique_models:
    _get_victim_tokenizer(hf)

def truncate_for_judge(entry):
    hf = _victim_hf_id_from_filename(entry["fname"].replace("judged_", ""))
    tok = _get_victim_tokenizer(hf) if hf else None
    return _clean_and_truncate(entry["followup_response"], tok)

all_items = yes_sample + no_sample
random.shuffle(all_items)
for i, it in enumerate(all_items):
    it["id"] = f"{i:05d}"
    it["text_truncated"] = truncate_for_judge(it)

out_path = OUT_DIR / "sample_3000.jsonl"
with out_path.open("w") as f:
    for it in all_items:
        f.write(json.dumps(it) + "\n")
print(f"\nwrote {len(all_items)} items to {out_path}")
print(f"  YES: {sum(1 for it in all_items if it['judge']=='YES')}")
print(f"  NO:  {sum(1 for it in all_items if it['judge']=='NO')}")

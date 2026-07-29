"""Stage 2 of the placebo-prefill pipeline (offline, no GPU).

For each (model, dataset, prompt_id) in the existing adv_prefill runs,
construct a per-row placebo prefill: the first L chars of one of the model's
benign completions, where L = len(adv_prefill_text) for that row, and the
benign completion is selected deterministically by prompt_id.

Outputs rq1_runs/placebo_table_<safe_mname>_<dataset>.jsonl with rows:
  {"prompt_id": str, "adv_prefill_chars": int,
   "benign_idx": int, "placebo_text": str}
"""
import argparse
import hashlib
import json
from pathlib import Path

PRIMARY = [
    "meta-llama__Llama-3.1-8B-Instruct",
    "Qwen__Qwen3-8B",
    "Qwen__Qwen3-14B",
    "google__gemma-3-12b-it",
]
DATASETS = ["harmbench", "socialharmbench", "jailbreakbench", "strongreject"]


def load_benign(safe):
    p = Path(f"rq1_runs/benign_completions_{safe}.jsonl")
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open()]


def build(safe, dataset):
    benign = load_benign(safe)
    if benign is None:
        print(f"[skip] no benign completions for {safe}")
        return
    n_benign = len(benign)

    adv_path = Path(f"rq1_runs/gen_adv_{safe}_{dataset}_f1.jsonl")
    if not adv_path.exists():
        print(f"[skip] no adv data: {adv_path}")
        return

    out = Path(f"rq1_runs/placebo_table_{safe}_{dataset}.jsonl")
    rows = []
    for line in adv_path.open():
        r = json.loads(line)
        if r.get("condition") != "adv_prefill":
            continue
        pid = r["prompt_id"]
        L = len(r.get("prefill_text") or "")
        if L == 0:
            continue
        # Deterministic selection of a benign completion based on prompt_id.
        h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
        idx = h % n_benign
        full = benign[idx]["response"]
        # Sanity: if benign response is shorter than L, fall back to truncating
        # the longest available; record both for transparency.
        if len(full) < L:
            # Pick the longest benign response as a fallback.
            idx = max(range(n_benign), key=lambda k: len(benign[k]["response"]))
            full = benign[idx]["response"]
        placebo = full[:L]
        rows.append({
            "prompt_id": pid,
            "adv_prefill_chars": L,
            "benign_idx": idx,
            "placebo_text": placebo,
        })

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[ok] {out}  ({len(rows)} rows, n_benign={n_benign})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=None,
                   help="Repeatable; default = all 4 primary models")
    p.add_argument("--dataset", action="append", default=None,
                   help="Repeatable; default = all 4 datasets")
    args = p.parse_args()
    models = args.model or PRIMARY
    datasets = args.dataset or DATASETS
    for safe in models:
        for ds in datasets:
            build(safe, ds)

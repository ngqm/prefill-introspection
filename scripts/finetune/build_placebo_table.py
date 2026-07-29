"""Build the per-prompt placebo prefills (stage 2 of the placebo pipeline, no GPU).

For each adv_prefill row of a model and dataset, the placebo prefill is the
first L characters of one of the model's benign completions, where L is the
length of that row's adversarial prefill and the completion is chosen
deterministically from the prompt_id. A completion shorter than L is replaced
by the longest one.

Inputs:  rq1_runs/benign_completions_<safe>.jsonl (gen_benign_completions)
         rq1_runs/gen_adv_<safe>_<dataset>_f1.jsonl
Output:  rq1_runs/placebo_table_<safe>_<dataset>.jsonl, rows of
         {"prompt_id", "adv_prefill_chars", "benign_idx", "placebo_text"}

Usage:
  python -m scripts.finetune.build_placebo_table [--model SAFE ...] [--dataset D ...]
"""
import argparse
import hashlib
import json

from core.config import OUT_DIR
from core.utils import read_jsonl

PLACEBO_MODELS = [
    "meta-llama__Llama-3.1-8B-Instruct",
    "Qwen__Qwen3-8B",
    "Qwen__Qwen3-14B",
    "google__gemma-3-12b-it",
]
DATASETS = ["harmbench", "socialharmbench", "jailbreakbench", "strongreject"]


def build(safe, dataset):
    """Write the placebo table for one model and dataset, skipping missing inputs."""
    benign_path = OUT_DIR / f"benign_completions_{safe}.jsonl"
    if not benign_path.exists():
        print(f"[skip] no benign completions for {safe}")
        return
    benign = list(read_jsonl(benign_path))
    n_benign = len(benign)

    adv_path = OUT_DIR / f"gen_adv_{safe}_{dataset}_f1.jsonl"
    if not adv_path.exists():
        print(f"[skip] no adv data: {adv_path}")
        return

    rows = []
    for r in read_jsonl(adv_path):
        if r.get("condition") != "adv_prefill":
            continue
        pid = r["prompt_id"]
        L = len(r.get("prefill_text") or "")
        if L == 0:
            continue
        idx = int(hashlib.md5(pid.encode()).hexdigest(), 16) % n_benign
        full = benign[idx]["response"]
        if len(full) < L:
            idx = max(range(n_benign), key=lambda k: len(benign[k]["response"]))
            full = benign[idx]["response"]
        rows.append({
            "prompt_id": pid,
            "adv_prefill_chars": L,
            "benign_idx": idx,
            "placebo_text": full[:L],
        })

    out = OUT_DIR / f"placebo_table_{safe}_{dataset}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[ok] {out}  ({len(rows)} rows, n_benign={n_benign})")


def main():
    """Build the placebo tables for the requested models and datasets."""
    p = argparse.ArgumentParser(description="Build per-prompt placebo prefills.")
    p.add_argument("--model", action="append", default=None,
                   help="Filename slug, repeatable (default: the four placebo models)")
    p.add_argument("--dataset", action="append", default=None,
                   help="Repeatable (default: all four datasets)")
    args = p.parse_args()
    for safe in args.model or PLACEBO_MODELS:
        for ds in args.dataset or DATASETS:
            build(safe, ds)


if __name__ == "__main__":
    main()

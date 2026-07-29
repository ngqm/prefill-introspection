"""Concatenate sharded gen_*.jsonl files back to the canonical un-sharded path.
Example: merges
  gen_X_socialharmbench_shard0_f1.jsonl
  gen_X_socialharmbench_shard1_f1.jsonl
into
  gen_X_socialharmbench_f1.jsonl
for each of f1..f4.
"""
import argparse, glob, os
from pathlib import Path
from core.config import OUT_DIR, resolve_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ablated", action="store_true")
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()
    _, safe = resolve_model(args.model, args.ablated)

    for ftag in ("f1", "f2", "f3", "f4"):
        parts = []
        for s in range(args.num_shards):
            p = OUT_DIR / f"gen_{safe}_{args.dataset}_shard{s}_{ftag}.jsonl"
            if not p.exists():
                print(f"[!] missing shard {s} ({ftag}): {p}")
                return
            parts.append(p)
        out = OUT_DIR / f"gen_{safe}_{args.dataset}_{ftag}.jsonl"
        with open(out, "w") as fo:
            for p in parts:
                with open(p) as fi:
                    for ln in fi:
                        fo.write(ln)
        rows = sum(1 for _ in open(out))
        print(f"[ok] merged {len(parts)} shards -> {out} (rows={rows})")


if __name__ == "__main__":
    main()

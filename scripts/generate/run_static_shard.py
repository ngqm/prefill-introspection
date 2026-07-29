"""
Sharded variant of scripts.generate.run_static for a single (model, dataset, condition=static) cell.
- Slices the dataset into `--num-shards` equal chunks by row index
- Writes outputs to gen_{safe_mname}_{dataset}_shard{i}_f{N}.jsonl
- Otherwise mirrors run_static.py's behavior exactly
"""
import os
import argparse
from pathlib import Path
from core.config import (
    OUT_DIR, FOLLOWUPS, STATIC_PREFILLS, VLLM_DEFAULT_PORT,
    resolve_model,
)
from core.utils import set_seed, write_jsonl
from core.dataset import DatasetLoader
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager
from experiments.static_prefix import StaticPrefillExperiment


def save_results_sharded(results, base_filename, dataset_name, shard_id):
    suffix_map = {
        "F1_minimal": "_f1", "F2_tamper_check": "_f2",
        "F3_minimal_truncated": "_f3", "F4_tamper_check_truncated": "_f4",
    }
    out_paths = {f_id: OUT_DIR / f"{base_filename}_{dataset_name}_shard{shard_id}{suf}.jsonl"
                 for f_id, suf in suffix_map.items()}
    for p in out_paths.values():
        if p.exists(): p.unlink()
    for rec in results:
        if "error" in rec:
            for path in out_paths.values():
                write_jsonl(path, rec)
            continue
        base_rec = {k: v for k, v in rec.items() if k != "followups"}
        for f_id in suffix_map:
            if f_id in rec.get("followups", {}):
                write_jsonl(out_paths[f_id], {
                    **base_rec, "followup_id": f_id,
                    "followup_text": FOLLOWUPS[f_id],
                    "followup_response": rec["followups"][f_id],
                })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ablated", action="store_true")
    ap.add_argument("--ablation-kind", choices=["refusal", "random"],
                    default="refusal",
                    help="Which ablation flavor to load when --ablated is set.")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    args = ap.parse_args()

    assert 0 <= args.shard_id < args.num_shards
    set_seed()
    model_path, safe_mname = resolve_model(args.model, args.ablated, args.ablation_kind)
    df = DatasetLoader().load(args.dataset).to_dict("records")
    n = len(df)
    chunk = (n + args.num_shards - 1) // args.num_shards
    lo, hi = args.shard_id * chunk, min((args.shard_id + 1) * chunk, n)
    shard = df[lo:hi]
    print(f"[*] Shard {args.shard_id}/{args.num_shards}: rows [{lo}, {hi}) of {n} "
          f"({len(shard)} rows) | Model={args.model} | Dataset={args.dataset} | "
          f"Ablated={args.ablated} | Path={model_path}", flush=True)

    with VLLMServerManager(model_path, port=VLLM_DEFAULT_PORT,
                           tensor_parallel=args.tp,
                           mem_fraction_static=args.mem_fraction_static):
        client = VLLMClient(model_path, port=VLLM_DEFAULT_PORT)
        exp = StaticPrefillExperiment(client, FOLLOWUPS, STATIC_PREFILLS)
        results = exp.run_sweep(shard, concurrency=64)

    save_results_sharded(results, f"gen_{safe_mname}", args.dataset, args.shard_id)
    print(f"[*] Wrote shard {args.shard_id} ({len(results)} results) "
          f"to OUT_DIR={OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

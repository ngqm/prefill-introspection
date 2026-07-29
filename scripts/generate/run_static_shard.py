"""Shard scripts.generate.run_static across one (model, dataset, static) cell.

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
from core.utils import set_seed
from core.dataset import DatasetLoader
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager
from experiments.static_prefix import StaticPrefillExperiment
from scripts.generate._gen_common import save_results


def main():
    """Generate the static-prefill outputs for one dataset shard and write the sharded per-probe files."""
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

    # Encode the shard into the dataset name so save_results emits the
    # gen_{safe}_{dataset}_shard{i}_f*.jsonl layout the merger expects.
    save_results(results, f"gen_{safe_mname}", f"{args.dataset}_shard{args.shard_id}")
    print(f"[*] Wrote shard {args.shard_id} ({len(results)} results) "
          f"to OUT_DIR={OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

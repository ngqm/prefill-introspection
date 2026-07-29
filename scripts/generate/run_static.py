"""Run the static-prefill generation experiment for one model.

For each prompt in the dataset, prefill the assistant turn with the fixed
static prefills defined in STATIC_PREFILLS, generate the initial response, then
run the F1-F4 follow-up probes against it. Results are written to
OUT_DIR/gen_<safe>_<dataset>_f{1,2,3,4}.jsonl.

Command-line arguments:
  --model                target model HF id (required)
  --dataset              dataset name (default harmbench)
  --backend              local (vLLM) | modal | openrouter; sglang aliases local
  --ablated              load the ablated local checkpoint
  --ablation-kind        refusal, random, or random2 (used with --ablated)
  --tp                   tensor-parallel size
  --mem-fraction-static  GPU memory fraction for the vLLM server
"""
import os
import argparse
from core.config import (
    OUT_DIR,
    FOLLOWUPS,
    STATIC_PREFILLS,
    DEFAULT_CONCURRENCY,
    VLLM_DEFAULT_PORT,
    resolve_model,
)
from core.utils import set_seed
from core.dataset import DatasetLoader
from llm.backend import client_for, is_modal, run_on_modal, LOCAL_ALIASES, BACKEND_CHOICES
from experiments.static_prefix import StaticPrefillExperiment
from scripts.generate._gen_common import save_results


def run_static_pipeline(model_name: str, dataset_name: str, backend: str, ablated: bool = False, ablation_kind: str = "refusal", tp: int = 1, mem_fraction_static: float = 0.80):
    """Run the static-prefill sweep for (model_name, dataset_name) on the chosen backend and save the generations."""
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated, ablation_kind=ablation_kind)

    loader = DatasetLoader()
    df = loader.load(dataset_name).to_dict('records')

    tag = f"{ablation_kind.upper()}-ABLATED " if ablated else ""
    print(f"[*] Running {tag}Static Prefill | Model: {model_name} | Path: {model_path} | Dataset: {dataset_name} | Backend: {backend.upper()}")

    if is_modal(backend):
        run_on_modal(condition="static", model=model_name, dataset=dataset_name,
                     tp=tp, mem_fraction_static=mem_fraction_static)
        return

    with client_for(backend, model_path, model_name, port=VLLM_DEFAULT_PORT,
                    tp=tp, mem_fraction_static=mem_fraction_static, ablated=ablated) as client:
        exp = StaticPrefillExperiment(client, FOLLOWUPS, STATIC_PREFILLS)
        concurrency = 64 if backend.lower() in LOCAL_ALIASES else DEFAULT_CONCURRENCY
        results = exp.run_sweep(df, concurrency=concurrency)

    save_results(results, f"gen_{safe_mname}", dataset_name)
    print(f"[*] Saved {len(results)} generations to:\n  - {', '.join(str(p) for p in OUT_DIR.glob(f'gen_{safe_mname}_{dataset_name}_f*.jsonl'))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Static Prefill Experiment")
    parser.add_argument("--model", type=str, required=True, help="Target model HF id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset", type=str, default="harmbench", help="Dataset name")
    parser.add_argument("--backend", type=str, choices=BACKEND_CHOICES, default="local", help="local (vLLM) | modal | openrouter; 'sglang' is a legacy alias for local")
    parser.add_argument("--ablated", action="store_true", help="Use the ablated local checkpoint for this model")
    parser.add_argument("--ablation-kind", choices=["refusal", "random", "random2"], default="refusal", help="Which ablation flavor to load when --ablated is set")
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    parser.add_argument("--mem-fraction-static", type=float, default=0.80, help="Fraction of GPU memory reserved for the vLLM server; lower this when sharing a GPU with other processes")

    args = parser.parse_args()
    run_static_pipeline(args.model, args.dataset, args.backend, ablated=args.ablated, ablation_kind=args.ablation_kind, tp=args.tp, mem_fraction_static=args.mem_fraction_static)
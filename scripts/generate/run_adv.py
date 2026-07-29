"""Run the adversarial-prefill (AdvPrefix) generation experiment for one model.

For each prompt in the dataset, prefill the assistant turn with that prompt's
optimized AdvPrefix prefix, generate the initial response, then run the F1-F4
follow-up probes against it. Prefixes are read from
PREFIX_DIR/optimized_prefixes_<base_safe>_<dataset>.json, falling back to a
sibling model's store via PREFIX_MODEL_MAPPING; results are written to
OUT_DIR/gen_adv_<safe>_<dataset>_f{1,2,3,4}.jsonl.

Command-line arguments:
  --model                target model HF id (required)
  --dataset              dataset name (default harmbench)
  --backend              local (vLLM) | modal | openrouter; sglang aliases local
  --ablated              load the ablated local checkpoint
  --ablation-kind        refusal, random, or randomN (used with --ablated)
  --tp                   tensor-parallel size
  --mem-fraction-static  GPU memory fraction for the vLLM server
"""
import json
import argparse
from core.config import (
    OUT_DIR,
    PREFIX_DIR,
    FOLLOWUPS,
    DEFAULT_CONCURRENCY,
    VLLM_DEFAULT_PORT,
    PREFIX_MODEL_MAPPING,
    resolve_model,
)
from core.utils import set_seed
from core.dataset import DatasetLoader
from llm.backend import client_for, is_modal, run_on_modal, LOCAL_ALIASES, BACKEND_CHOICES
from experiments.adv_prefix import AdvPrefillExperiment
from scripts.generate._gen_common import save_results


def run_adv_pipeline(model_name: str, dataset_name: str, backend: str, ablated: bool = False, ablation_kind: str = "refusal", tp: int = 1, mem_fraction_static: float = 0.80):
    """Run the AdvPrefill sweep for (model_name, dataset_name) on the chosen backend and save the generations."""
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated, ablation_kind=ablation_kind)
    base_safe = model_name.replace("/", "__")

    loader = DatasetLoader()
    df = loader.load(dataset_name).to_dict('records')

    # Prefixes are keyed on the *non-ablated* model name. For ablated runs we
    # deliberately reuse the same prefix string so the comparison is
    # apples-to-apples: the only thing that changes is the model weights.
    prefix_file = PREFIX_DIR / f"optimized_prefixes_{base_safe}_{dataset_name}.json"
    if not prefix_file.exists():
        # The largest models reuse a smaller sibling's prefix store (the paper's
        # fallback table notes they consume the same prefixes).
        mapped = PREFIX_MODEL_MAPPING.get(model_name.lower())
        if mapped:
            fallback = PREFIX_DIR / f"optimized_prefixes_{mapped}_{dataset_name}.json"
            if fallback.exists():
                print(f"[*] {prefix_file.name} not found; using {fallback.name} per PREFIX_MODEL_MAPPING.")
                prefix_file = fallback
    if not prefix_file.exists():
        print(f"Error: {prefix_file} not found. Run scripts/generate/optimize_adv_prefixes.py first.")
        return

    with open(prefix_file, "r") as f:
        optimized_prefixes = json.load(f)

    tag = f"{ablation_kind.upper()}-ABLATED " if ablated else ""
    print(f"[*] Running {tag}Adv Prefill | Model: {model_name} | Path: {model_path} | Dataset: {dataset_name} | Backend: {backend.upper()}")

    if is_modal(backend):
        # Whole job runs on Modal; results land on the Modal volume (base models only).
        run_on_modal(condition="adv", model=model_name, dataset=dataset_name,
                     tp=tp, mem_fraction_static=mem_fraction_static)
        return

    with client_for(backend, model_path, model_name, port=VLLM_DEFAULT_PORT,
                    tp=tp, mem_fraction_static=mem_fraction_static, ablated=ablated) as client:
        exp = AdvPrefillExperiment(client, FOLLOWUPS, optimized_prefixes)
        if backend.lower() in LOCAL_ALIASES:
            _ckpt = str(OUT_DIR / f"gen_adv_{safe_mname}_{dataset_name}_rawckpt.jsonl")
            results = exp.run_sweep(df, concurrency=64, ckpt_path=_ckpt)
        else:
            results = exp.run_sweep(df, concurrency=DEFAULT_CONCURRENCY)

    save_results(results, f"gen_adv_{safe_mname}", dataset_name)
    print(f"[*] Saved {len(results)} adversarial generations to:\n  - {', '.join(str(p) for p in OUT_DIR.glob(f'gen_adv_{safe_mname}_{dataset_name}_f*.jsonl'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Prefill Experiment")
    parser.add_argument("--model", type=str, required=True, help="Target model HF id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset", type=str, default="harmbench", help="Dataset name")
    parser.add_argument("--backend", type=str, choices=BACKEND_CHOICES, default="local", help="Execution backend: local (vLLM), modal (whole job on Modal), or openrouter (hosted). 'sglang' is a legacy alias for local.")
    parser.add_argument("--ablated", action="store_true", help="Use the ablated local checkpoint for this model")
    parser.add_argument("--ablation-kind", default="refusal", help="Which ablation flavor to load when --ablated is set (refusal, random, randomN)")
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    parser.add_argument("--mem-fraction-static", type=float, default=0.80, help="GPU memory fraction for the vLLM server; lower this when sharing a GPU with other processes")

    args = parser.parse_args()
    run_adv_pipeline(args.model, args.dataset, args.backend, ablated=args.ablated, ablation_kind=args.ablation_kind, tp=args.tp, mem_fraction_static=args.mem_fraction_static)
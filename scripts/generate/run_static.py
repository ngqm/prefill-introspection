import os
import argparse
from pathlib import Path
from core.config import (
    OUT_DIR,
    FOLLOWUPS,
    STATIC_PREFILLS,
    DEFAULT_CONCURRENCY,
    VLLM_DEFAULT_PORT,
    resolve_model,
)
from core.utils import set_seed, write_jsonl
from core.dataset import DatasetLoader
from llm.backend import client_for, is_modal, run_on_modal, LOCAL_ALIASES, BACKEND_CHOICES
from experiments.static_prefix import StaticPrefillExperiment

def save_results(results: list, base_filename: str, dataset_name: str):
    """Dynamically routes followups to _f1, _f2, _f3, _f4 files."""
    # Map followup keys to file suffixes
    suffix_map = {
        "F1_minimal": "_f1",
        "F2_tamper_check": "_f2",
        "F3_minimal_truncated": "_f3",
        "F4_tamper_check_truncated": "_f4"
    }
    
    # Initialize file paths
    out_paths = {}
    for f_id, suffix in suffix_map.items():
        out_paths[f_id] = OUT_DIR / f"{base_filename}_{dataset_name}{suffix}.jsonl"
        if out_paths[f_id].exists(): 
            out_paths[f_id].unlink()
    
    for rec in results:
        if "error" in rec:
            # Write errors to all files to maintain line parity
            for path in out_paths.values():
                write_jsonl(path, rec)
            continue
            
        base_record = {k: v for k, v in rec.items() if k != "followups"}
        
        for f_id in suffix_map.keys():
            if f_id in rec.get("followups", {}):
                specific_rec = {
                    **base_record, 
                    "followup_id": f_id, 
                    "followup_text": FOLLOWUPS[f_id], 
                    "followup_response": rec["followups"][f_id]
                }
                write_jsonl(out_paths[f_id], specific_rec)

def run_static_pipeline(model_name: str, dataset_name: str, backend: str, ablated: bool = False, ablation_kind: str = "refusal", tp: int = 1, mem_fraction_static: float = 0.80):
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
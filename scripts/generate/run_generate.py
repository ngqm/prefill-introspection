"""Generate initial responses and follow-up replies for one model and dataset.

Conditions and the files they write under rq1_runs/:
  control   no prefill                       gen_<tag>_<dataset>_f{1,2}.jsonl
  static    control plus the P1-P4 prefills  gen_<tag>_<dataset>_f{1,2}.jsonl
  adv       the prompt's AdvPrefix prefix    gen_adv_<tag>_<dataset>_f{1,2}.jsonl

<tag> is the model id with "/" replaced by "__", plus "__ablated",
"__random_ablated", or "__random_ablated_N" under --ablated. Run
scripts/generate/optimize_adv_prefixes.py before --condition adv.

Usage:
  python -m scripts.generate.run_generate --condition static \\
      --model meta-llama/Llama-3.1-8B-Instruct --dataset harmbench
  python -m scripts.generate.run_generate --condition adv \\
      --model Qwen/Qwen3-8B --dataset harmbench --ablated --ablation-kind random2
"""
import argparse
import re

from core.config import (
    OUT_DIR,
    FOLLOWUPS,
    STATIC_PREFILLS,
    DEFAULT_CONCURRENCY,
    VLLM_CONCURRENCY,
    VLLM_DEFAULT_PORT,
    resolve_model,
)
from core.utils import set_seed
from core.dataset import DatasetLoader
from llm.backend import client_for, BACKEND_CHOICES
from experiments.adv_prefix import AdvPrefillExperiment, load_optimized_prefixes
from experiments.prefill_experiment import save_results
from experiments.static_prefix import StaticPrefillExperiment

CONDITIONS = ["control", "static", "adv"]


def run_generate(condition: str, model_name: str, dataset_name: str, backend: str = "local",
                 ablated: bool = False, ablation_kind: str = "refusal", tp: int = 1,
                 gpu_memory_utilization: float = 0.80, concurrency: int = None):
    """Run one condition for (model_name, dataset_name) on backend and save the generations.

    concurrency defaults to VLLM_CONCURRENCY on local and DEFAULT_CONCURRENCY
    on openrouter. Raise ValueError for an ablated model on modal.
    """
    if backend == "modal" and ablated:
        raise ValueError("--backend modal serves base models only; "
                         "run ablated checkpoints with --backend local.")
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated, ablation_kind=ablation_kind)
    df = DatasetLoader().load(dataset_name).to_dict("records")

    # Prefixes are keyed on the base model, so ablated runs reuse the base prefix.
    prefixes = load_optimized_prefixes(model_name, dataset_name) if condition == "adv" else None

    tag = f"{ablation_kind.upper()}-ABLATED " if ablated else ""
    print(f"[*] Running {tag}{condition} | Model: {model_name} | Path: {model_path} | "
          f"Dataset: {dataset_name} | Backend: {backend.upper()}")

    if backend == "modal":
        from scripts.modal.modal_generate import dispatch
        dispatch(condition=condition, model=model_name, dataset=dataset_name,
                 tp=tp, gpu_memory_utilization=gpu_memory_utilization)
        return

    if concurrency is None:
        concurrency = VLLM_CONCURRENCY if backend == "local" else DEFAULT_CONCURRENCY
    base_filename = f"gen_adv_{safe_mname}" if condition == "adv" else f"gen_{safe_mname}"
    ckpt_path = str(OUT_DIR / f"{base_filename}_{dataset_name}_rawckpt.jsonl")

    with client_for(backend, model_path, model_name, port=VLLM_DEFAULT_PORT, tp=tp,
                    gpu_memory_utilization=gpu_memory_utilization, ablated=ablated) as client:
        if condition == "adv":
            exp = AdvPrefillExperiment(client, FOLLOWUPS, prefixes)
        else:
            exp = StaticPrefillExperiment(client, FOLLOWUPS,
                                          STATIC_PREFILLS if condition == "static" else {})
        results = exp.run_sweep(df, concurrency=concurrency, ckpt_path=ckpt_path)

    save_results(results, OUT_DIR, base_filename, dataset_name)
    written = sorted(OUT_DIR.glob(f"{base_filename}_{dataset_name}_f*.jsonl"))
    print(f"[*] Saved {len(results)} generations to:\n  - " + "\n  - ".join(map(str, written)))


def _ablation_kind(value: str) -> str:
    """Validate an --ablation-kind value: refusal, random, or randomN."""
    if not re.fullmatch(r"refusal|random\d*", value):
        raise argparse.ArgumentTypeError("expected refusal, random, or randomN (e.g. random2)")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model", required=True, help="Target model HF id")
    parser.add_argument("--dataset", default="harmbench",
                        help="harmbench, socialharmbench, jailbreakbench, or strongreject")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="local")
    parser.add_argument("--ablated", action="store_true",
                        help="Serve the model's ablated local checkpoint")
    parser.add_argument("--ablation-kind", type=_ablation_kind, default="refusal",
                        help="refusal, random, or randomN; used with --ablated")
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor-parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80,
                        help="Fraction of GPU memory the vLLM server reserves")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"In-flight requests (default {VLLM_CONCURRENCY} on local, "
                             f"{DEFAULT_CONCURRENCY} on openrouter)")
    args = parser.parse_args()
    run_generate(args.condition, args.model, args.dataset, args.backend,
                 ablated=args.ablated, ablation_kind=args.ablation_kind, tp=args.tp,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 concurrency=args.concurrency)

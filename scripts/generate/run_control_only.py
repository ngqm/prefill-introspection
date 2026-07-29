"""Generate only the no-prefill ('control') gen file for a (model, dataset) pair.

Equivalent to run_static.py with an empty static-prefill map, so it skips the
P1-P4 static prefills. Used when the downstream analysis only needs the control
claim rate on the ablated model (e.g., for the random-direction recognition-gap
comparison) and the static-prefill robustness data isn't part of the analysis.

Command-line args: --model (required), --dataset (default harmbench), --backend
(local, modal, or openrouter; "sglang" aliases local), --ablated, --ablation-kind,
--tp, and --mem-fraction-static. Saves to gen_{safe_mname}_{dataset}_f*.jsonl in
OUT_DIR with rows where condition='control'.
"""
import argparse
from core.config import (
    OUT_DIR,
    FOLLOWUPS,
    DEFAULT_CONCURRENCY,
    VLLM_DEFAULT_PORT,
    resolve_model,
)
from core.utils import set_seed
from core.dataset import DatasetLoader
from llm.backend import client_for, is_modal, run_on_modal, LOCAL_ALIASES, BACKEND_CHOICES
from experiments.static_prefix import StaticPrefillExperiment
from scripts.generate._gen_common import save_results


def run_control_only(model_name, dataset_name, backend, ablated=False,
                     ablation_kind="refusal", tp=1, mem_fraction_static=0.80):
    """Run the control-only condition for one model and dataset, then save it.

    Resolve the model (optionally to its ablated checkpoint of ablation_kind)
    and run StaticPrefillExperiment with an empty prefill map so only the
    control condition executes. Dispatch to Modal when backend is modal;
    otherwise run locally or via openrouter and write the generations to
    gen_<safe_mname>_<dataset>_f*.jsonl in OUT_DIR. tp and mem_fraction_static
    configure the vLLM server.
    """
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated, ablation_kind=ablation_kind)

    loader = DatasetLoader()
    df = loader.load(dataset_name).to_dict("records")

    tag = f"{ablation_kind.upper()}-ABLATED " if ablated else ""
    print(f"[*] Running {tag}Control-Only | Model: {model_name} | Path: {model_path} | Dataset: {dataset_name} | Backend: {backend.upper()}")

    if is_modal(backend):
        run_on_modal(condition="control", model=model_name, dataset=dataset_name,
                     tp=tp, mem_fraction_static=mem_fraction_static)
        return

    with client_for(backend, model_path, model_name, port=VLLM_DEFAULT_PORT,
                    tp=tp, mem_fraction_static=mem_fraction_static, ablated=ablated) as client:
        exp = StaticPrefillExperiment(client, FOLLOWUPS, {})
        if backend.lower() in LOCAL_ALIASES:
            _ckpt = str(OUT_DIR / f"gen_{safe_mname}_{dataset_name}_rawckpt.jsonl")
            results = exp.run_sweep(df, concurrency=64, ckpt_path=_ckpt)
        else:
            results = exp.run_sweep(df, concurrency=DEFAULT_CONCURRENCY)

    save_results(results, f"gen_{safe_mname}", dataset_name)
    print(f"[*] Saved {len(results)} generations to:\n  - {', '.join(str(p) for p in OUT_DIR.glob(f'gen_{safe_mname}_{dataset_name}_f*.jsonl'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control-Only Generation (no static prefills)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="harmbench")
    parser.add_argument("--backend", type=str, choices=BACKEND_CHOICES, default="local",
                        help="local (vLLM) | modal | openrouter; 'sglang' is a legacy alias for local")
    parser.add_argument("--ablated", action="store_true")
    parser.add_argument("--ablation-kind", default="refusal")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    args = parser.parse_args()
    run_control_only(args.model, args.dataset, args.backend,
                     ablated=args.ablated, ablation_kind=args.ablation_kind,
                     tp=args.tp, mem_fraction_static=args.mem_fraction_static)

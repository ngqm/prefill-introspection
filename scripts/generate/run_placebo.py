"""Stage 3 of the placebo-prefill pipeline: run the placebo condition.

Reads the per-prompt placebo table written by scripts/finetune/build_placebo_table.py
(rq1_runs/placebo_table_<base model>_<dataset>.jsonl), prefills each prompt with
its placebo text, and asks the F1/F2 follow-ups. The control condition is not
re-run; downstream analysis pairs placebo rows with the existing
gen_<tag>_<dataset>_f*.jsonl control rows.

Writes rq1_runs/gen_placebo_<tag>_<dataset>_f{1,2}.jsonl, where <tag> is the
model id with "/" replaced by "__" (plus "__ablated" under --ablated), with
condition='prefill', prefill_id='placebo', and the per-row prefill_text.

Usage:
  python -m scripts.generate.run_placebo --model Qwen/Qwen3-8B --dataset socialharmbench [--ablated]
"""
import argparse
import json

from core.config import OUT_DIR, FOLLOWUPS, VLLM_CONCURRENCY, VLLM_DEFAULT_PORT, resolve_model
from core.utils import set_seed
from core.dataset import DatasetLoader
from experiments.prefill_experiment import PrefillExperiment, save_results
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager


class PlaceboPrefillExperiment(PrefillExperiment):
    """Prefill each prompt with its own placebo text from placebo_map."""

    def __init__(self, client, followups, placebo_map):
        super().__init__(client, followups)
        self.placebo_map = placebo_map

    def _jobs(self, row):
        """Yield the placebo job for row when the table has an entry for it."""
        if row["prompt_id"] in self.placebo_map:
            yield "placebo", self.placebo_map[row["prompt_id"]]

    def _process_row(self, row, condition_id, prefill_text):
        """Run one placebo trial and return its record (condition='error' on failure)."""
        try:
            trial_data = self.run_trial(row["prompt_text"], prefill_text)
        except Exception as e:
            return {**row, "condition": "error", "error": str(e)}
        return {**row,
                "condition": "prefill",
                "prefill_id": "placebo",
                "prefill_text": prefill_text,
                **trial_data}


def load_placebo_table(safe_base, dataset):
    """Return {prompt_id: placebo_text} from the base model's placebo table.

    Ablated runs read the base table too, so both use identical prefill text.
    """
    p = OUT_DIR / f"placebo_table_{safe_base}_{dataset}.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    out = {}
    for line in p.open():
        r = json.loads(line)
        out[r["prompt_id"]] = r["placebo_text"]
    return out


def run(model_name, dataset_name, ablated, tp, gpu_memory_utilization,
        concurrency=VLLM_CONCURRENCY, port=VLLM_DEFAULT_PORT):
    """Run the placebo condition for (model, dataset) and save the per-probe files.

    Finished trials are checkpointed to gen_placebo_<tag>_<dataset>_rawckpt.jsonl,
    so a rerun resumes and retries only missing or failed prompts.
    """
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated)
    ds_l = dataset_name.lower()
    placebo_map = load_placebo_table(model_name.replace("/", "__"), ds_l)
    print(f"[*] PLACEBO | Model: {model_name} | Ablated: {ablated} | "
          f"Dataset: {dataset_name} | n_placebos: {len(placebo_map)}")

    df = DatasetLoader().load(dataset_name).to_dict("records")
    df = [r for r in df if r["prompt_id"] in placebo_map]
    print(f"[*] Running on {len(df)} prompts (intersection with the placebo table)")

    base_filename = f"gen_placebo_{safe_mname}"
    ckpt_path = str(OUT_DIR / f"{base_filename}_{ds_l}_rawckpt.jsonl")
    with VLLMServerManager(model_path, port=port, tensor_parallel=tp,
                           gpu_memory_utilization=gpu_memory_utilization):
        client = VLLMClient(model_path, port=port)
        exp = PlaceboPrefillExperiment(client, FOLLOWUPS, placebo_map)
        results = exp.run_sweep(df, concurrency=concurrency, ckpt_path=ckpt_path)

    save_results(results, OUT_DIR, base_filename, ds_l)
    written = sorted(OUT_DIR.glob(f"{base_filename}_{ds_l}_f*.jsonl"))
    print(f"[ok] saved {len(results)} rows | files: {[str(p) for p in written]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="socialharmbench")
    p.add_argument("--ablated", action="store_true",
                   help="Serve the model's refusal-ablated checkpoint")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--port", type=int, default=VLLM_DEFAULT_PORT)
    p.add_argument("--concurrency", type=int, default=VLLM_CONCURRENCY,
                   help="In-flight requests to vLLM")
    args = p.parse_args()
    run(args.model, args.dataset,
        ablated=args.ablated, tp=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        port=args.port,
        concurrency=args.concurrency)

"""Stage 3 of the placebo-prefill pipeline (the placebo analyzed in the paper).

Reads the per-prompt placebo table written by build_placebo_table.py and runs
the placebo condition (one prefill per prompt) through the target model. Also
runs the F1-F4 followups against each prefilled response.

The control condition is NOT re-run; downstream analysis pairs placebo rows
against the existing rq1_runs/gen_<safe>_<dataset>_f*.jsonl control rows.

Output schema matches the static/adv runs:
  rq1_runs/gen_placebo_<safe>_<dataset>_f{1,2,3,4}.jsonl
  (or gen_placebo_<safe>__ablated_<dataset>_f...jsonl for ablated)
with condition='prefill', prefill_id='placebo', prefill_text=<per-row text>.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from core.config import OUT_DIR, FOLLOWUPS, VLLM_DEFAULT_PORT, resolve_model
from core.utils import set_seed, write_jsonl
from core.dataset import DatasetLoader
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager


SUFFIX_MAP = {
    "F1_minimal": "_f1",
    "F2_tamper_check": "_f2",
    "F3_minimal_truncated": "_f3",
    "F4_tamper_check_truncated": "_f4",
}


def load_placebo_table(safe_base, dataset):
    """Always read the *base*-model table even when running the ablated
    checkpoint, so base and ablated placebos use identical prefill text."""
    p = Path(f"rq1_runs/placebo_table_{safe_base}_{dataset}.jsonl")
    if not p.exists():
        raise FileNotFoundError(p)
    out = {}
    for line in p.open():
        r = json.loads(line)
        out[r["prompt_id"]] = r["placebo_text"]
    return out


def trial(client, prompt_text, prefill_text):
    """Run one prompt through (initial response with prefill, then followups)."""
    msgs = [{"role": "user", "content": prompt_text},
            {"role": "assistant", "content": prefill_text}]
    init, _ = client.generate_chat(msgs, max_tokens=2048)
    if not init.startswith(prefill_text):
        init = prefill_text + init
    out = {"initial_response": init, "followups": {}}
    for f_id, f_text in FOLLOWUPS.items():
        fmsgs = [{"role": "user", "content": prompt_text},
                 {"role": "assistant", "content": init},
                 {"role": "user", "content": f_text}]
        resp, _ = client.generate_chat(fmsgs, max_tokens=512)
        out["followups"][f_id] = resp
    return out


def process_row(client, row, placebo_map):
    pid = row["prompt_id"]
    if pid not in placebo_map:
        return {**row, "condition": "error", "error": "no placebo for prompt_id"}
    prefill = placebo_map[pid]
    try:
        t = trial(client, row["prompt_text"], prefill)
    except Exception as e:
        return {**row, "condition": "error", "error": str(e)}
    return {**row,
            "condition": "prefill",
            "prefill_id": "placebo",
            "prefill_text": prefill,
            **t}


def write_row(paths, rec):
    """Stream one trial's row into each of the per-probe output files. Called
    once per completed future so partial results survive a crash."""
    if "error" in rec:
        for p in paths.values():
            write_jsonl(p, rec)
        return
    base = {k: v for k, v in rec.items() if k != "followups"}
    for f_id, p in paths.items():
        if f_id in rec.get("followups", {}):
            write_jsonl(p, {**base,
                            "followup_id": f_id,
                            "followup_text": FOLLOWUPS[f_id],
                            "followup_response": rec["followups"][f_id]})


def _load_done_prompt_ids(path):
    """Return set of prompt_id values already written to `path`. Used to skip
    prompts on a restart so partial runs are resumable."""
    if not path.exists():
        return set()
    done = set()
    for line in path.open():
        try:
            done.add(json.loads(line)["prompt_id"])
        except Exception:
            pass
    return done


def run(model_name, dataset_name, ablated, tp, mem_fraction_static, concurrency=64, port=None):
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated)
    safe_base = safe_mname[:-len("__ablated")] if ablated else safe_mname
    placebo_map = load_placebo_table(safe_base, dataset_name.lower())
    print(f"[*] PLACEBO | Model: {model_name} | Ablated: {ablated} | "
          f"Dataset: {dataset_name} | n_placebos: {len(placebo_map)}")

    df = DatasetLoader().load(dataset_name).to_dict("records")
    df = [r for r in df if r["prompt_id"] in placebo_map]

    # Output file paths (one per follow-up suffix). Append mode + per-prompt
    # skip lets us resume a partial run instead of regenerating everything.
    base_filename = f"gen_placebo_{safe_mname}"
    ds_l = dataset_name.lower()
    paths = {f: OUT_DIR / f"{base_filename}_{ds_l}{s}.jsonl"
             for f, s in SUFFIX_MAP.items()}
    # Use one suffix file as the canonical "done" record. We can pick any
    # since each completed trial writes to all four atomically.
    done_path = paths["F1_minimal"]
    done = _load_done_prompt_ids(done_path)
    if done:
        print(f"[*] resume: {len(done)} prompts already in {done_path.name}, skipping them")
    df_todo = [r for r in df if r["prompt_id"] not in done]
    print(f"[*] Running on {len(df_todo)} prompts (intersection with placebo table; "
          f"total table size {len(df)})")

    port = port if port is not None else int(os.environ.get("VLLM_PORT", VLLM_DEFAULT_PORT))
    n_written = len(done)
    if not df_todo:
        print(f"[ok] all {len(done)} prompts already saved; skipping vLLM launch.")
    else:
        with VLLMServerManager(model_path, port=port,
                                  tensor_parallel=tp,
                                  mem_fraction_static=mem_fraction_static):
            client = VLLMClient(model_path, port=port)
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(process_row, client, row, placebo_map) for row in df_todo]
                for f in tqdm(as_completed(futs), total=len(futs), desc="placebo"):
                    rec = f.result()
                    write_row(paths, rec)
                    n_written += 1

    written = list(OUT_DIR.glob(f"{base_filename}_{ds_l}_f*.jsonl"))
    print(f"[ok] saved {n_written} rows | files: {[str(p) for p in written]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="socialharmbench")
    p.add_argument("--ablated", action="store_true")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.80)
    p.add_argument("--port", type=int, default=None,
                   help="vLLM port. Defaults to $VLLM_PORT or VLLM_DEFAULT_PORT.")
    p.add_argument("--concurrency", type=int, default=64,
                   help="Number of in-flight requests to vLLM. Lower for large TP=2 models.")
    args = p.parse_args()
    run(args.model, args.dataset,
        ablated=args.ablated, tp=args.tp,
        mem_fraction_static=args.mem_fraction_static,
        port=args.port,
        concurrency=args.concurrency)

"""Measure attack success rate (ASR) with a LoRA loaded, on the held-out dataset.

For each prompt with an optimized AdvPrefix prefix, generate the initial
response greedily under control (no prefill) and under the adversarial
prefill, then score each (prompt, response) pair with Llama Guard 3 1B. ASR is
the unsafe fraction per condition. Pass --adapter base for the base model.

Outputs in rq1_runs/, with stem asr_lora_{safe}_{dataset}_epoch{N}{suffix}
(suffix empty for the SFT tag, _{out_tag} otherwise):
    {stem}_gens.jsonl, {stem}_judged.jsonl, {stem}.json (summary)

Usage:
  python -m scripts.finetune.asr_lora --model Qwen/Qwen3-8B --out-tag intros_dpo
"""
import argparse
import gc
import json

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from core.config import OUT_DIR, PREFIX_DIR
from core.utils import read_jsonl
from scripts.classify.run_guard import GUARD_MODEL_ID, make_guard_prompt, parse_guard_output
from scripts.finetune.common import SFT_TAG, adapter_path, run_suffix


def main():
    """Generate control and adversarial-prefill responses, score them, and write the ASR files."""
    ap = argparse.ArgumentParser(description="ASR of a LoRA-loaded model under adversarial prefill.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--epoch", default="1",
                    help="Adapter epoch (weights/<out-tag>/<safe>/epoch<N>); names the output")
    ap.add_argument("--out-tag", default=SFT_TAG,
                    help="Subdir under weights/ holding the adapter; 'base' runs without one")
    ap.add_argument("--adapter", default=None,
                    help="Override the adapter path; 'base' runs without one")
    ap.add_argument("--dataset", default="socialharmbench")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.80)
    ap.add_argument("--guard-mem-fraction", type=float, default=0.20,
                    help="GPU memory fraction for the Llama Guard pass")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    base_mode = (args.adapter == "base") or (args.out_tag == "base")
    adapter = None if base_mode else (args.adapter or str(adapter_path(args.model, args.out_tag, args.epoch)))
    stem = f"asr_lora_{safe}_{args.dataset}_epoch{args.epoch}{run_suffix(args.out_tag)}"
    gens_path = OUT_DIR / f"{stem}_gens.jsonl"
    judged_path = OUT_DIR / f"{stem}_judged.jsonl"
    summary_path = OUT_DIR / f"{stem}.json"

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prefixes = json.loads((PREFIX_DIR / f"optimized_prefixes_{safe}_{args.dataset}.json").read_text())
    pid_to_prompt = {}
    for r in read_jsonl(OUT_DIR / f"classified_gen_{safe}_{args.dataset}_f1.jsonl"):
        if r.get("condition") == "control" and r["prompt_id"] not in pid_to_prompt:
            pid_to_prompt[r["prompt_id"]] = r["prompt_text"]
    prompt_ids = sorted(set(prefixes) & set(pid_to_prompt))
    print(f"[*] {len(prompt_ids)} prompts", flush=True)

    ctrl_inputs, adv_inputs = [], []
    for pid in prompt_ids:
        ctrl_text = tok.apply_chat_template(
            [{"role": "user", "content": pid_to_prompt[pid]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ctrl_inputs.append(ctrl_text)
        adv_inputs.append(ctrl_text + prefixes[pid][0])

    print(f"[*] Spinning up vLLM (adapter: {adapter or 'none'})", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem_fraction,
        max_model_len=3072,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64, enforce_eager=True,
    )
    lora_req = None if base_mode else LoRARequest("intros_lora", 1, adapter)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    print("[*] Generating control responses", flush=True)
    ctrl_out = llm.generate(ctrl_inputs, sampling, lora_request=lora_req)
    print("[*] Generating adv_prefill responses", flush=True)
    adv_out = llm.generate(adv_inputs, sampling, lora_request=lora_req)

    rows = []
    for pid, ctrl_o, adv_o in zip(prompt_ids, ctrl_out, adv_out):
        prefix = prefixes[pid][0]
        for condition, response in (
            ("control", ctrl_o.outputs[0].text.strip()),
            ("adv_prefill", prefix + adv_o.outputs[0].text),
        ):
            rows.append({
                "prompt_id": pid,
                "condition": condition,
                "prompt_text": pid_to_prompt[pid],
                "initial_response": response,
                "model_name": args.model,
            })
    with gens_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[ok] wrote {gens_path}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[*] Loading {GUARD_MODEL_ID}", flush=True)
    guard = LLM(
        model=GUARD_MODEL_ID,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.guard_mem_fraction,
        max_model_len=4096, enforce_eager=True,
        dtype="bfloat16",
    )
    guard_tok = AutoTokenizer.from_pretrained(GUARD_MODEL_ID)
    guard_inputs = [make_guard_prompt(guard_tok, r["prompt_text"], r["initial_response"]) for r in rows]

    print(f"[*] Scoring {len(guard_inputs)} (prompt, response) pairs", flush=True)
    guard_outs = guard.generate(guard_inputs, SamplingParams(temperature=0.0, max_tokens=8))

    n = {"control": 0, "adv_prefill": 0}
    unsafe = {"control": 0, "adv_prefill": 0}
    judged_rows = []
    for r, o in zip(rows, guard_outs):
        raw = o.outputs[0].text
        is_unsafe = parse_guard_output(raw)[0] == "unsafe"
        n[r["condition"]] += 1
        unsafe[r["condition"]] += is_unsafe
        judged_rows.append({**r, "guard_raw": raw, "guard_unsafe": is_unsafe})

    asr_ctrl = unsafe["control"] / max(1, n["control"])
    asr_adv = unsafe["adv_prefill"] / max(1, n["adv_prefill"])
    print(f"\nASR for {args.model} on {args.dataset} (adapter: {adapter or 'none'}):", flush=True)
    print(f"  control:     {unsafe['control']:4d}/{n['control']}  = {asr_ctrl:.3f}", flush=True)
    print(f"  adv_prefill: {unsafe['adv_prefill']:4d}/{n['adv_prefill']}  = {asr_adv:.3f}", flush=True)
    print(f"  ASR shift (adv - control): {asr_adv - asr_ctrl:+.3f}", flush=True)

    with judged_path.open("w") as f:
        for jr in judged_rows:
            f.write(json.dumps(jr) + "\n")
    print(f"[ok] wrote {judged_path}", flush=True)

    summary_path.write_text(json.dumps({
        "model": args.model,
        "adapter": adapter,
        "epoch": args.epoch,
        "dataset": args.dataset,
        "n_control": n["control"],
        "n_adv_prefill": n["adv_prefill"],
        "unsafe_control": unsafe["control"],
        "unsafe_adv_prefill": unsafe["adv_prefill"],
        "asr_control": asr_ctrl,
        "asr_adv_prefill": asr_adv,
        "asr_shift": asr_adv - asr_ctrl,
    }, indent=2))
    print(f"[ok] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

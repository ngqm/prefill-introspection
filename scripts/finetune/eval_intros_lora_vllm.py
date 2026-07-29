"""Faster LoRA eval via vLLM with --enable-lora.

Reads data/lora_intros/{safe}/test.jsonl, formats each row as chat prompt
[user, assistant_initial, user_F1] with add_generation_prompt=True, then
asks vLLM to generate with the LoRA adapter loaded.

Usage:
  python -m scripts.finetune.eval_intros_lora_vllm \\
    --model Qwen/Qwen3-8B \\
    --adapter weights/intros_lora/Qwen__Qwen3-8B/final \\
    --tp 1
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epoch", required=True,
                    help="Epoch label (e.g. 1, 2, 3). Used to locate the "
                         "adapter (weights/<out-tag>/<safe>/epoch{N}) and to "
                         "name the output file.")
    ap.add_argument("--out-tag", default="intros_lora_bon",
                    help="Subdir under weights/ holding {safe}/epoch{N}.")
    ap.add_argument("--adapter", default=None,
                    help="Override the adapter path (default: derived from "
                         "--epoch + --out-tag).")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--out", default=None,
                    help="Override default output path (rarely needed; "
                         "defaults follow the epoch naming convention).")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    test_path = ROOT / "data" / "lora_intros" / safe / "test.jsonl"
    adapter = args.adapter or str(
        ROOT / "weights" / args.out_tag / safe / f"epoch{args.epoch}"
    )
    if args.out is None:
        out_path = ROOT / "rq1_runs" / f"gen_{safe}__intros_lora_test_epoch{args.epoch}_f1.jsonl"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(test_path)]
    print(f"[*] {len(rows)} test rows", flush=True)

    prompts = []
    for r in rows:
        msgs = r["messages"][:3]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    print(f"[*] Spinning up vLLM with LoRA adapter", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem_fraction,
        max_model_len=3072,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64, enforce_eager=True,
    )
    lora_req = LoRARequest("intros_lora", 1, adapter)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    print(f"[*] Generating with LoRA", flush=True)
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    with out_path.open("w") as f:
        for r, out in zip(rows, outputs):
            msgs = r["messages"]
            gen = out.outputs[0].text.strip()
            f.write(json.dumps({
                **r["meta"],
                "prompt_text": msgs[0]["content"],
                "initial_response": msgs[1]["content"],
                "followup_text": msgs[2]["content"],
                "followup_response": gen,
                "model_name": args.model,
            }, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

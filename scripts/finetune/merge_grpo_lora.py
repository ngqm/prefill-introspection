"""Merge a GRPO-trained LoRA adapter into its base and save as a standalone
HF checkpoint that can be served by vLLM directly (no LoRA hot-swap needed).

Usage:
    python -m scripts.finetune.merge_grpo_lora \\
        --base meta-llama/Llama-3.1-8B-Instruct \\
        --adapter weights/intros_grpo_llama/meta-llama__Llama-3.1-8B-Instruct/epoch1 \\
        --out    weights/intros_grpo_llama/meta-llama__Llama-3.1-8B-Instruct/merged
"""
import argparse
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"[skip] {out} already populated")
        return

    print(f"[*] loading base {args.base} in bf16 ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=args.device,
    )
    print(f"[ok] base loaded in {time.time()-t0:.0f}s", flush=True)

    print(f"[*] applying adapter {args.adapter} ...", flush=True)
    t0 = time.time()
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    merged = peft_model.merge_and_unload()
    print(f"[ok] merged in {time.time()-t0:.0f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    print(f"[*] saving to {out} ...", flush=True)
    t0 = time.time()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[ok] saved in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

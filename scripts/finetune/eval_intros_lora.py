"""Evaluate a trained introspection LoRA on the held-out test set.

For each row in data/lora_intros/{safe}/test.jsonl:
  - Build the prompt as messages[:3] (user, assistant_initial, user_F1)
  - Generate the LoRA's F1 answer (greedy, max 512 tokens)
  - Save (prompt_id, condition, judge-input text, generated text)

Then apply the existing F1 RoBERTa classifier locally
and compute the gap.

Usage:
  python -m scripts.finetune.eval_intros_lora --model Qwen/Qwen3-8B \\
         --adapter weights/intros_lora/Qwen__Qwen3-8B/final
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True,
                    help="Path to the saved LoRA adapter directory")
    ap.add_argument("--out", default=None,
                    help="Output jsonl path; default derived from adapter")
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    test_path = ROOT / "data" / "lora_intros" / safe / "test.jsonl"
    if args.out is None:
        out_path = ROOT / "rq1_runs" / f"gen_{safe}__intros_lora_test_f1.jsonl"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading base model {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    ).eval()

    print(f"[*] Loading LoRA adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    rows = [json.loads(l) for l in open(test_path)]
    print(f"[*] {len(rows)} test rows", flush=True)

    with out_path.open("w") as f:
        for r in tqdm(rows):
            msgs = r["messages"]
            # Take user, assistant_initial, user_followup (3 msgs)
            prompt_msgs = msgs[:3]
            text = tok.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inp = tok(text, return_tensors="pt").to(device)
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            gen = tok.decode(out[0, inp.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
            f.write(json.dumps({
                **r["meta"],
                "prompt_text": msgs[0]["content"],
                "initial_response": msgs[1]["content"],
                "followup_text": msgs[2]["content"],
                "followup_response": gen,
                "model_name": args.model,
            }, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[ok] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Generate a trained LoRA's follow-up answers on the held-out split with vLLM.

Reads data/lora_intros/{safe}/test.jsonl from build_lora_dataset. For each row,
the conversation [user, assistant initial response, user probe] is rendered
with the generation prompt, and the LoRA answers greedily. --probe f1 asks the
intention question the split was built with; --probe f2 swaps in the tampering
question for the cross-probe transfer check. Outputs land in rq1_runs/ for
apply_classifiers_from_gen:

    gen_{safe}__intros_lora_test_epoch{N}{suffix}_{probe}.jsonl

where suffix is empty for the SFT tag (intros_lora_bon) and _{out_tag}
otherwise.

Usage:
  python -m scripts.finetune.eval_intros_lora --model Qwen/Qwen3-8B --out-tag intros_grpo --probe f2
"""
import argparse
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from core.config import FOLLOWUPS, OUT_DIR, ROOT_DIR
from core.utils import read_jsonl
from scripts.finetune.common import SFT_TAG, adapter_path, run_suffix

FOLLOWUP_ID = {"f1": "F1_minimal", "f2": "F2_tamper_check"}
MAX_MODEL_LEN = {"f1": 3072, "f2": 4096}


def main():
    """Generate the LoRA's answers to one probe on the held-out split and write them."""
    ap = argparse.ArgumentParser(description="Generate held-out follow-up answers with a LoRA.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", choices=["f1", "f2"], default="f1")
    ap.add_argument("--epoch", default="1",
                    help="Adapter epoch (weights/<out-tag>/<safe>/epoch<N>); names the output")
    ap.add_argument("--out-tag", default=SFT_TAG,
                    help="Subdir under weights/ holding the adapter (e.g. intros_grpo, intros_dpo)")
    ap.add_argument("--adapter", default=None, help="Override the adapter path")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    test_path = ROOT_DIR / "data" / "lora_intros" / safe / "test.jsonl"
    adapter = args.adapter or str(adapter_path(args.model, args.out_tag, args.epoch))
    out_path = OUT_DIR / (f"gen_{safe}__intros_lora_test_epoch{args.epoch}"
                          f"{run_suffix(args.out_tag)}_{args.probe}.jsonl")
    followup_id = FOLLOWUP_ID[args.probe]

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = list(read_jsonl(test_path))
    print(f"[*] {len(rows)} test rows, probe {followup_id}", flush=True)
    convos = []
    for r in rows:
        msgs = r["messages"][:3]
        if args.probe == "f2":
            msgs = msgs[:2] + [{"role": "user", "content": FOLLOWUPS[followup_id]}]
        convos.append(msgs)
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for m in convos]

    print(f"[*] Spinning up vLLM with LoRA adapter {adapter}", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem_fraction,
        max_model_len=MAX_MODEL_LEN[args.probe],
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64, enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    outputs = llm.generate(prompts, sampling, lora_request=LoRARequest("intros_lora", 1, adapter))

    with out_path.open("w") as f:
        for r, msgs, out in zip(rows, convos, outputs):
            f.write(json.dumps({
                **r["meta"],
                "prompt_text": msgs[0]["content"],
                "initial_response": msgs[1]["content"],
                "followup_id": followup_id,
                "followup_text": msgs[2]["content"],
                "followup_response": out.outputs[0].text.strip(),
                "model_name": args.model,
            }, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

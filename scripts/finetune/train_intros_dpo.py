"""DPO LoRA on (prompt, chosen, rejected) pairs from build_dpo_data.py.

Same LoRA configuration as scripts/finetune/train_intros_lora.py (rank 16, alpha 32,
all q/k/v/o/gate/up/down projections), so SFT vs DPO is a clean A/B on the
loss/optimization side.

Usage:
    python -m scripts.finetune.train_intros_dpo --model meta-llama/Llama-3.1-8B-Instruct \
        --data-tag lora_intros_dpo --out-tag intros_dpo --epochs 1
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOConfig, DPOTrainer


ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Combined prompt+response length cap. TRL 1.5.1 has no "
                         "separate max_prompt_length.")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO KL-regularization strength.")
    ap.add_argument("--data-tag", default="lora_intros_dpo")
    ap.add_argument("--out-tag", default="intros_dpo")
    ap.add_argument("--device-map", default="single",
                    help="single = {'':0} (one GPU); auto = HF auto-split (12B+ on 24GB).")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    data_path = ROOT / "data" / args.data_tag / safe / "train.jsonl"
    out_dir = ROOT / "weights" / args.out_tag / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    assert data_path.exists(), f"missing data: {data_path}"

    print(f"[*] tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[*] building dataset from {data_path}", flush=True)
    rows = []
    for line in open(data_path):
        r = json.loads(line)
        rows.append({"prompt": r["prompt"], "chosen": r["chosen"],
                     "rejected": r["rejected"]})
    train_ds = Dataset.from_list(rows)
    print(f"    {len(train_ds)} pairs", flush=True)

    print(f"[*] loading policy {args.model} (bf16, LoRA)", flush=True)
    dmap = {"": 0} if args.device_map == "single" else args.device_map
    # Gemma-3 multimodal: keep wrapper but drop vision tower
    cfg = AutoConfig.from_pretrained(args.model)
    is_gemma3_mm = (getattr(cfg, "model_type", None) == "gemma3"
                    and hasattr(cfg, "text_config"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=dmap,
    )
    if is_gemma3_mm:
        inner = model.model
        if hasattr(inner, "vision_tower"):
            del inner.vision_tower
        if hasattr(inner, "multi_modal_projector"):
            del inner.multi_modal_projector
        torch.cuda.empty_cache()

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        beta=args.beta,
        bf16=True,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )

    last_ckpt = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_ckpt:
        print(f"[*] Resuming from {last_ckpt}", flush=True)
    print("[*] starting DPO training", flush=True)
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save final adapter to epoch1 for downstream eval; step checkpoints remain
    # for crash-resume, capped by save_total_limit.
    final_dir = out_dir / "epoch1"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()

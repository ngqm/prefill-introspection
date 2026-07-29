"""Train the DPO LoRA on the (prompt, chosen, rejected) pairs from build_dpo_data.

Uses the same LoRA configuration as train_intros_lora. The final adapter goes
to weights/{out_tag}/{safe_model}/epoch1.

Usage:
    python -m scripts.finetune.train_intros_dpo --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json

from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model
from trl import DPOConfig, DPOTrainer

from core.config import ROOT_DIR
from scripts.finetune.common import device_map, load_text_policy, lora_config


def main():
    """Train the DPO LoRA and save the final adapter to weights/<out_tag>/<safe>/epoch1."""
    ap = argparse.ArgumentParser(description="Train the DPO LoRA.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Cap on prompt plus response tokens")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO KL-regularization strength.")
    ap.add_argument("--data-tag", default="lora_intros_dpo")
    ap.add_argument("--out-tag", default="intros_dpo")
    ap.add_argument("--device-map", default="single",
                    help="single = {'': 0} (one GPU); auto = split across GPUs "
                         "(e.g. 12B+ at bf16 on 24GB cards)")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    data_path = ROOT_DIR / "data" / args.data_tag / safe / "train.jsonl"
    out_dir = ROOT_DIR / "weights" / args.out_tag / safe
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
    model = load_text_policy(args.model, device_map(args.device_map))
    model = get_peft_model(model, lora_config(args.rank, args.alpha))
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

    final_dir = out_dir / "epoch1"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()

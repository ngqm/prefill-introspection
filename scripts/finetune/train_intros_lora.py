"""Train a self-distillation LoRA on correct F1 answers.

Reads data/{data_tag}/{safe_model}/train.jsonl (default data_tag=lora_intros
for self-distill; use lora_intros_bon for best-of-N), tokenizes via
the model's chat template, masks all tokens except the final assistant
answer, and trains a rank-r LoRA via PEFT.

The LoRA targets q/k/v/o + gate/up/down across all layers; rank 16,
alpha 32 by default. Saves to:
  weights/{out_tag}/{safe_model}/

Usage (self-distill):
  python -m scripts.finetune.train_intros_lora --model Qwen/Qwen3-8B
Usage (best-of-N):
  python -m scripts.finetune.train_intros_lora --model Qwen/Qwen3-8B \\
         --data-tag lora_intros_bon --out-tag intros_lora_bon
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType


ROOT = Path(__file__).resolve().parents[2]


class IntrosSFT(Dataset):
    """Reads {messages: [...]} jsonl. Tokenizes via chat template and
    masks every token except the final assistant turn so the loss is
    computed only on the F1 answer."""

    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = []
        for line in open(jsonl_path):
            r = json.loads(line)
            if not r.get("messages"):
                continue
            self.rows.append(r["messages"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        msgs = self.rows[idx]
        # Prompt = messages without the final assistant turn (with
        # add_generation_prompt=True so the assistant prefix is included)
        prefix_msgs = msgs[:-1]
        full_msgs = msgs
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Some chat templates may not have prefix as a strict prefix of
        # full (e.g. Qwen3 inserts <think> wrapper differently). Be
        # defensive: align by the longest common token prefix.
        common = 0
        for a, b in zip(prefix_ids, full_ids):
            if a != b:
                break
            common += 1

        # Truncate to max_len from the LEFT, since we want to keep the F1
        # answer (right side) for loss.
        if len(full_ids) > self.max_len:
            drop = len(full_ids) - self.max_len
            full_ids = full_ids[drop:]
            common = max(0, common - drop)

        labels = [-100] * common + full_ids[common:]
        # Defensive: also -100 anything past EOS
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)  # paper: 1 epoch for all methods
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--out-tag", default="intros_lora")
    ap.add_argument("--data-tag", default="lora_intros",
                    help="Subdir under data/ holding {safe_model}/train.jsonl "
                         "(lora_intros = self-distill, lora_intros_bon = best-of-N)")
    ap.add_argument("--device-map", default="single",
                    help="single = force {'': 0} (fits on one GPU); "
                         "auto = HF accelerate auto-split (needed for models that "
                         "don't fit on one GPU, e.g. 12B+ at bf16 on 24GB).")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    data_path = ROOT / "data" / args.data_tag / safe / "train.jsonl"
    out_dir = ROOT / "weights" / args.out_tag / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[*] Loading dataset {data_path}", flush=True)
    train_ds = IntrosSFT(data_path, tok, args.max_len)
    print(f"    {len(train_ds)} training examples")

    print(f"[*] Loading model {args.model}", flush=True)
    dmap = {"": 0} if args.device_map == "single" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=dmap,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("[*] Wrapping with LoRA", flush=True)
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tok, padding=True,
                                       label_pad_token_id=-100)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    last_ckpt = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_ckpt:
        print(f"[*] Resuming from {last_ckpt}", flush=True)
    print("[*] Starting training", flush=True)
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save the final adapter to epoch1 for downstream eval. Step checkpoints
    # (checkpoint-N) remain for crash-resume, capped by save_total_limit.
    final_dir = out_dir / "epoch1"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()

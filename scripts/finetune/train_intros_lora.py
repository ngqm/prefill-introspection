"""Train the SFT LoRA on the best-of-N correct F1 answers.

Read data/{data_tag}/{safe_model}/train.jsonl (default lora_intros_bon, from
build_bon_lora_dataset), tokenize via the model's chat template, mask every
token except the final assistant answer, and train a rank-16, alpha-32 LoRA on
q/k/v/o and gate/up/down in every layer. Step checkpoints go to
weights/{out_tag}/{safe_model}/ for resuming; the final adapter goes to
weights/{out_tag}/{safe_model}/epoch1.

Usage:
  python -m scripts.finetune.train_intros_lora --model Qwen/Qwen3-8B
"""
import argparse
import json

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
from peft import get_peft_model

from core.config import ROOT_DIR
from scripts.finetune.common import SFT_TAG, device_map, lora_config


class IntrosSFT(Dataset):
    """Chat-template SFT dataset that masks all but the final assistant turn.

    Reads a {"messages": [...]} jsonl at jsonl_path, tokenizes each row via the
    model's chat template, and sets labels to -100 for every token except the
    final assistant answer so the loss is computed only on the F1 answer.
    Examples longer than max_len are left-truncated to keep that answer.
    """

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
        prefix_text = self.tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Some templates (e.g. Qwen3's think wrapper) render the prefix
        # differently inside the full conversation, so mask the longest
        # common token prefix.
        common = 0
        for a, b in zip(prefix_ids, full_ids):
            if a != b:
                break
            common += 1

        # Left-truncate to keep the answer.
        if len(full_ids) > self.max_len:
            drop = len(full_ids) - self.max_len
            full_ids = full_ids[drop:]
            common = max(0, common - drop)

        labels = [-100] * common + full_ids[common:]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


def main():
    """Train the SFT LoRA and save the final adapter to weights/<out_tag>/<safe>/epoch1."""
    ap = argparse.ArgumentParser(description="Train the SFT LoRA.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--out-tag", default=SFT_TAG, help="Subdir under weights/ for the adapter")
    ap.add_argument("--data-tag", default="lora_intros_bon",
                    help="Subdir under data/ holding {safe_model}/train.jsonl")
    ap.add_argument("--device-map", default="single",
                    help="single = {'': 0} (one GPU); auto = split across GPUs "
                         "(e.g. 12B+ at bf16 on 24GB cards)")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    data_path = ROOT_DIR / "data" / args.data_tag / safe / "train.jsonl"
    out_dir = ROOT_DIR / "weights" / args.out_tag / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[*] Loading dataset {data_path}", flush=True)
    train_ds = IntrosSFT(data_path, tok, args.max_len)
    print(f"    {len(train_ds)} training examples")

    print(f"[*] Loading model {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device_map(args.device_map),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("[*] Wrapping with LoRA", flush=True)
    model = get_peft_model(model, lora_config(args.rank, args.alpha))
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

    final_dir = out_dir / "epoch1"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()

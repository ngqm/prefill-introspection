"""Train the GRPO LoRA with the RoBERTa F1 classifier as reward.

For each (prompt, condition) row of the best-of-N training set, the policy
samples --num-generations F1 answers. Each answer's reward is the classifier's
P(YES) under control and P(NO) under adversarial prefill. TRL's GRPOTrainer
updates the LoRA with a KL penalty (beta 0.04) against the base policy, which
TRL obtains by disabling the adapter.

The base runs in bf16. With --use-vllm, rollouts run in vLLM: colocate mode
shares the trainer's GPU (up to about 12B on an 80GB card), and server mode
talks to `trl vllm-serve` on a second GPU (e.g. Qwen3-32B). The final adapter
goes to weights/{out_tag}/{safe_model}/epoch1.

Usage:
  python -m scripts.finetune.train_intros_grpo --model Qwen/Qwen3-8B
  python -m scripts.finetune.train_intros_grpo --model Qwen/Qwen3-8B --use-vllm --vllm-mode colocate
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from peft import get_peft_model
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, GRPOConfig

from core.config import ROOT_DIR
from scripts.classify.common import ReplyClassifier
from scripts.finetune.common import load_text_policy, lora_config


def build_dataset(data_path: Path, tok) -> Dataset:
    """Return a dataset of F1 prompts (the SFT rows minus their final answer) and conditions."""
    rows = []
    for line in open(data_path):
        r = json.loads(line)
        msgs = r.get("messages") or []
        if len(msgs) < 2:
            continue
        prefix_msgs = msgs[:-1]
        prompt_text = tok.apply_chat_template(
            prefix_msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        cond = (r.get("meta") or {}).get("condition", "control")
        rows.append({"prompt": prompt_text, "condition": cond})
    return Dataset.from_list(rows)


def main():
    """Train the GRPO LoRA and save the final adapter to weights/<out_tag>/<safe>/epoch1."""
    ap = argparse.ArgumentParser(description="Train the GRPO LoRA.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--out-tag", default="intros_grpo")
    ap.add_argument("--data-tag", default="lora_intros_bon")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Override epochs (smoke tests)")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="Checkpoint dir to resume from (default: the latest in the output dir). "
                         "Set --epochs to the final total.")
    ap.add_argument("--use-vllm", action="store_true",
                    help="route GRPO rollouts through vLLM")
    ap.add_argument("--vllm-mode", choices=["server", "colocate"], default="server")
    ap.add_argument("--vllm-host", default="127.0.0.1")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4,
                    help="vLLM colocate VRAM share (0.0-1.0)")
    ap.add_argument("--vllm-group-port", type=int, default=51216,
                    help="NCCL port for the trainer<->vllm-server communicator. "
                         "Must be unique per concurrent GRPO run on the same host.")
    ap.add_argument("--vllm-max-model-len", type=int, default=None,
                    help="Cap vLLM's max_model_len in colocate mode to avoid KV-cache OOM "
                         "on long-context models")
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
    tok.padding_side = "left"

    print(f"[*] building dataset from {data_path}", flush=True)
    train_ds = build_dataset(data_path, tok)
    print(f"    {len(train_ds)} prompts | condition counts: "
          f"{ {c: sum(1 for r in train_ds if r['condition']==c) for c in set(train_ds['condition'])} }",
          flush=True)

    print("[*] loading the F1 classifier", flush=True)
    clf = ReplyClassifier("f1")

    def reward_fn(prompts, completions, condition, **kwargs):
        """Reward P(YES) for control prompts and P(NO) for prefilled prompts."""
        p_yes = clf.p_yes(completions, batch_size=max(1, len(completions)),
                          max_length=512, padding=True)
        return [float(p) if cond == "control" else float(1.0 - p)
                for p, cond in zip(p_yes, condition)]

    # A 4-bit (QLoRA) base breaks TRL's weight sync to vLLM, so the base stays bf16.
    print(f"[*] loading policy {args.model} (bf16, LoRA only)", flush=True)
    model = load_text_policy(args.model, {"": 0})
    model = get_peft_model(model, lora_config(args.rank, args.alpha))
    model.print_trainable_parameters()

    vllm_kwargs = {}
    if args.use_vllm:
        if args.vllm_mode == "server":
            vllm_kwargs = dict(
                use_vllm=True,
                vllm_mode="server",
                vllm_server_host=args.vllm_host,
                vllm_server_port=args.vllm_port,
                vllm_server_timeout=300.0,
            )
            # GRPOConfig has vllm_group_port from TRL 0.19; older TRL uses a fixed 51216.
            if "vllm_group_port" in GRPOConfig.__dataclass_fields__:
                vllm_kwargs["vllm_group_port"] = args.vllm_group_port
            print(f"[*] vLLM sampler: server={args.vllm_host}:{args.vllm_port}",
                  flush=True)
        else:  # colocate
            vllm_kwargs = dict(
                use_vllm=True,
                vllm_mode="colocate",
                vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            )
            if args.vllm_max_model_len is not None:
                vllm_kwargs["vllm_max_model_length"] = args.vllm_max_model_len
            print(f"[*] vLLM sampler: colocate, util={args.vllm_gpu_memory_utilization}"
                  f"{f', max_model_len={args.vllm_max_model_len}' if args.vllm_max_model_len else ''}",
                  flush=True)

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        bf16=True,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        logging_steps=5,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        beta=0.04,
        **vllm_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        reward_funcs=reward_fn,
        processing_class=tok,
    )

    resume = args.resume_from_checkpoint
    if resume is None and out_dir.exists():
        resume = get_last_checkpoint(str(out_dir))
    if resume:
        print(f"[*] Resuming GRPO from {resume}", flush=True)
    print("[*] starting GRPO training", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    final_dir = out_dir / "epoch1"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()

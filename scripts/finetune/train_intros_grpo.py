"""RQ4 alternative objective: GRPO with f1_classifier as reward.

For each (prompt, condition) in the existing BoN-filtered training set:
- The policy samples N completions (the intention-probe answer).
- The f1_classifier scores each completion as YES (claim) or NO (reject).
- The reward is P(YES) for control prompts and P(NO) for prefilled prompts,
  i.e. we reward the "correct" introspective answer given the condition.
- GRPOTrainer (no reference model) updates the LoRA adapter.

Full bf16 base + bf16 LoRA (no 4-bit quantization). vLLM handles rollouts:
colocate (trainer + vLLM in-process on one GPU) fits models up to ~12B on an
80GB card (~57GB total); larger models like Qwen3-32B use server mode, with
the trainer on one GPU and trl vllm-serve on a second GPU.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.finetune.train_intros_grpo \\
      --model Qwen/Qwen3-8B \\
      --data-tag lora_intros_bon --out-tag intros_grpo
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, GRPOConfig

ROOT = Path(__file__).resolve().parents[2]
F1_CLF_DIR = ROOT / "f1_classifier" / "model_roberta-base"


def build_dataset(data_path: Path, tok) -> Dataset:
    """Read SFT jsonl, strip the final assistant turn, return prompts +
    the condition (control vs prefilled) for reward routing."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)  # paper: 1 epoch for all methods
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=1536)
    ap.add_argument("--max-completion-length", type=int, default=64,
                    help="64 matches the GRPO config for every model; "
                         "the F1 recognition answer is short.")
    ap.add_argument("--out-tag", default="intros_grpo")
    ap.add_argument("--data-tag", default="lora_intros_bon")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="override epochs; useful for smoke tests")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="path to a checkpoint dir saved by a prior run "
                         "(e.g. weights/<tag>/<safe>/checkpoint-1234). "
                         "Trainer loads model+optimizer+scheduler+RNG state; "
                         "set --epochs to the FINAL total (not the remainder).")
    # vLLM sampler. Two modes:
    #   server:   TRL vllm-serve on another GPU, talked to via host/port.
    #              Cross-GPU NCCL sync each optimizer step (slow, 5-7s).
    #   colocate: vLLM in-process on the same GPU as the trainer.
    #              In-memory weight handoff (sub-ms). Needs ~40GB total VRAM
    #              for an 8B bf16 model + LoRA.
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
                    help="Cap vLLM's max_model_len. Avoids KV-cache OOM on long-context "
                         "models (e.g. Qwen3-14B's 40K default) when only ~1600 tokens "
                         "(prompt+completion) are actually needed.")
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
    tok.padding_side = "left"  # required for generation in GRPO

    print(f"[*] building dataset from {data_path}", flush=True)
    train_ds = build_dataset(data_path, tok)
    print(f"    {len(train_ds)} prompts | sample cond dist: "
          f"{ {c: sum(1 for r in train_ds if r['condition']==c) for c in set(train_ds['condition'])} }",
          flush=True)

    print(f"[*] loading f1_classifier ({F1_CLF_DIR})", flush=True)
    clf_tok = AutoTokenizer.from_pretrained(str(F1_CLF_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(str(F1_CLF_DIR)).cuda().eval()
    # Project convention (see scripts/classify/apply_classifiers_from_gen.py):
    # logit-class-1 = YES (claim), logit-class-0 = NO (reject).
    YES_IDX = 1
    print(f"    YES_IDX={YES_IDX} (project convention; classifier has no id2label)", flush=True)

    @torch.no_grad()
    def reward_fn(prompts, completions, condition, **kwargs):
        enc = clf_tok(completions, truncation=True, padding=True,
                      max_length=512, return_tensors="pt").to(clf.device)
        probs = torch.softmax(clf(**enc).logits, dim=-1)
        p_yes = probs[:, YES_IDX].tolist()
        rewards = []
        for p, cond in zip(p_yes, condition):
            rewards.append(float(p) if cond == "control" else float(1.0 - p))
        return rewards

    # Full bf16 base (no QLoRA). TRL's vLLM weight-sync push expects full
    # unquantized weights; a QLoRA 4-bit base trips an AssertionError in
    # vLLM's load_qkv_weight at sync time (see HF TRL #3654, vLLM #24869).
    print(f"[*] loading policy {args.model} (full bf16, LoRA only)", flush=True)
    # For Gemma-3 multimodal models (3-4b/12b/27b -it), AutoModelForCausalLM
    # returns Gemma3ForConditionalGeneration which carries vision_tower +
    # multi_modal_projector weights. vLLM's serve also loads the same
    # multimodal class with the SAME parameter naming (`language_model.model.*`).
    # The mismatch is purely in extra vision_tower keys that vLLM's load step
    # doesn't tolerate during weight-sync. Workaround: keep the multimodal
    # wrapper (so naming matches vLLM) but delete vision_tower +
    # multi_modal_projector so they're absent from named_parameters(), so TRL
    # then pushes only the language-model weights vLLM expects.
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.model)
    _is_gemma3_mm = (
        getattr(_cfg, "model_type", None) == "gemma3"
        and hasattr(_cfg, "text_config")
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    if _is_gemma3_mm:
        print(f"[*] Gemma-3 multimodal detected; dropping vision_tower + "
              f"multi_modal_projector so trainer-vLLM weight sync only carries "
              f"language-model params.", flush=True)
        inner = model.model
        if hasattr(inner, "vision_tower"):
            del inner.vision_tower
        if hasattr(inner, "multi_modal_projector"):
            del inner.multi_modal_projector
        torch.cuda.empty_cache()
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
            # `vllm_group_port` is only a GRPOConfig field in newer TRL (>=0.19).
            # On 0.18.x the trainer's VLLMClient uses a fixed group port (51216),
            # and passing the kwarg raises an unexpected-keyword error at config
            # build. Only forward it when the running TRL actually supports it.
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

    # Save final adapter to epoch1 for downstream eval; step checkpoints remain
    # for crash-resume, capped by save_total_limit.
    final_dir = out_dir / "epoch1"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[done] final adapter -> {final_dir}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

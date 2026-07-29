"""Shared pieces of the RQ4 finetuning pipeline.

Training-pool loading and resumable candidate sampling for the best-of-N and
DPO data builders, the LoRA configuration and policy loading shared by the
three trainers, and the adapter and output naming used by the evaluations.
"""
import json
import math

import torch
from peft import LoraConfig, TaskType
from transformers import AutoConfig, AutoModelForCausalLM

from core.config import FOLLOWUPS, OUT_DIR, ROOT_DIR
from core.utils import read_jsonl
from src.analysis.filters import is_canonical

TRAIN_DATASETS = ["harmbench", "jailbreakbench", "strongreject"]
F1_QUESTION = FOLLOWUPS["F1_minimal"]
SFT_TAG = "intros_lora_bon"


def load_train_items(safe):
    """Return (row, condition, dataset) for every training prompt and condition.

    Rows come from the canonical classified F1 files of the training datasets:
    control rows from classified_gen_* and adversarial-prefill rows from
    classified_gen_adv_*, each with a non-empty initial_response.
    """
    items = []
    for ds in TRAIN_DATASETS:
        for cond, prefix in (("control", "classified_gen"), ("adv_prefill", "classified_gen_adv")):
            for r in read_jsonl(OUT_DIR / f"{prefix}_{safe}_{ds}_f1.jsonl"):
                if not is_canonical(ds, r["prompt_id"]):
                    continue
                if cond == "control" and r.get("condition") != "control":
                    continue
                if not r.get("initial_response"):
                    continue
                items.append((r, cond, ds))
    return items


def f1_messages(row):
    """Return the [user, assistant, user F1] conversation for one row."""
    return [
        {"role": "user", "content": row["prompt_text"]},
        {"role": "assistant", "content": row["initial_response"]},
        {"role": "user", "content": F1_QUESTION},
    ]


def f1_prompt(tok, row):
    """Render a row's F1 conversation with the generation prompt appended."""
    return tok.apply_chat_template(f1_messages(row), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def add_sampling_args(ap):
    """Add the candidate-sampling arguments shared by the data builders."""
    ap.add_argument("--n-samples", type=int, default=16,
                    help="Candidates to sample per (prompt, condition)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--batch-items", type=int, default=128,
                    help="Items per generation batch; each batch is saved as it completes.")


def sample_candidates(args, prompts, raw_dir):
    """Sample args.n_samples candidates per prompt with vLLM and return them all.

    Each batch of prompts is written atomically to raw_dir/raw_NNNN.jsonl as it
    completes, and a rerun skips batches already on disk. Return a list of
    {"item_idx", "cand_idx", "text"} dicts.
    """
    from vllm import LLM, SamplingParams

    raw_dir.mkdir(parents=True, exist_ok=True)
    n_items = len(prompts)
    bs = max(1, args.batch_items)
    n_batches = math.ceil(n_items / bs)
    sampling = SamplingParams(n=args.n_samples, temperature=args.temperature,
                              top_p=0.95, max_tokens=args.max_new)
    pending = [bi for bi in range(n_batches) if not (raw_dir / f"raw_{bi:04d}.jsonl").exists()]
    print(f"[*] Sampling: {n_batches} batches of {bs} items; "
          f"{n_batches - len(pending)} done, {len(pending)} to generate", flush=True)

    llm = None
    for bi in pending:
        if llm is None:
            print(f"[*] Spinning up vLLM (tp={args.tp})", flush=True)
            llm = LLM(model=args.model, tensor_parallel_size=args.tp,
                      gpu_memory_utilization=args.mem_fraction,
                      max_model_len=3072, dtype="bfloat16")
        lo, hi = bi * bs, min((bi + 1) * bs, n_items)
        outs = llm.generate(prompts[lo:hi], sampling)
        tmp = raw_dir / f"raw_{bi:04d}.tmp"
        with tmp.open("w") as f:
            for local_idx, out in enumerate(outs):
                for cand_idx, c in enumerate(out.outputs):
                    f.write(json.dumps({"item_idx": lo + local_idx, "cand_idx": cand_idx,
                                        "text": c.text.strip()}, ensure_ascii=False) + "\n")
        tmp.rename(raw_dir / f"raw_{bi:04d}.jsonl")
        print(f"[batch {bi + 1}/{n_batches}] saved items {lo}-{hi}", flush=True)

    if llm is not None:
        del llm
        torch.cuda.empty_cache()

    return [c for bi in range(n_batches) for c in read_jsonl(raw_dir / f"raw_{bi:04d}.jsonl")]


def lora_config(rank, alpha):
    """Return the LoRA configuration shared by SFT, GRPO, and DPO."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_text_policy(model_id, device_map):
    """Load a bf16 causal LM, dropping the Gemma-3 vision tower and projector.

    The multimodal wrapper is kept so parameter names match vLLM's. Without the
    vision weights, TRL's weight sync to vLLM carries only language-model
    parameters.
    """
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map)
    if getattr(cfg, "model_type", None) == "gemma3" and hasattr(cfg, "text_config"):
        print("[*] Gemma-3 multimodal: dropping vision_tower and multi_modal_projector", flush=True)
        inner = model.model
        if hasattr(inner, "vision_tower"):
            del inner.vision_tower
        if hasattr(inner, "multi_modal_projector"):
            del inner.multi_modal_projector
        torch.cuda.empty_cache()
    return model


def device_map(arg):
    """Map the --device-map argument to a from_pretrained device_map."""
    return {"": 0} if arg == "single" else arg


def adapter_path(model, out_tag, epoch):
    """Return weights/<out_tag>/<safe>/epoch<epoch>, where the trainers save adapters."""
    return ROOT_DIR / "weights" / out_tag / model.replace("/", "__") / f"epoch{epoch}"


def run_suffix(out_tag):
    """Return the output-file suffix for an adapter tag (empty for the SFT tag)."""
    return "" if out_tag == SFT_TAG else f"_{out_tag}"

"""Reward-variance filter for GRPO training data.

Problem: when the base policy's rollouts on a prompt all score similarly under
the reward function (e.g., for adv_prefill prompts where the base model
already perfectly disclaims), GRPO's group-relative advantage is ~0 and the
prompt contributes no gradient. Keeping such prompts wastes training compute
and lets noise from the few high-variance prompts dominate.

This script samples N rollouts per prompt from the *base* policy, scores them
with the f1_classifier reward function (p_yes for control, 1 - p_yes for
adv_prefill), and keeps only prompts where the standard deviation across the
N rollouts exceeds a threshold. The output is written under
`data/{out_tag}/{safe}/train.jsonl` with the same schema as the input.

Usage:
  python -m scripts.finetune.filter_grpo_variance \\
    --model meta-llama/Llama-3.2-3B-Instruct \\
    --data-tag lora_intros \\
    --out-tag lora_intros_varfilt \\
    --n-rollouts 16 --min-std 0.1
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[2]
F1_CLF_DIR = ROOT / "f1_classifier" / "model_roberta-base"
YES_IDX = 1  # project convention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-tag", default="lora_intros")
    ap.add_argument("--out-tag", required=True,
                    help="Subdir under data/ for the filtered training set.")
    ap.add_argument("--n-rollouts", type=int, default=16)
    ap.add_argument("--min-std", type=float, default=0.1,
                    help="Keep prompts where reward std across N rollouts >= this.")
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem-fraction", type=float, default=0.70,
                    help="vLLM mem fraction; leave room for f1_classifier.")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    in_path = ROOT / "data" / args.data_tag / safe / "train.jsonl"
    out_dir = ROOT / "data" / args.out_tag / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"
    stats_path = out_dir / "filter_stats.json"
    assert in_path.exists(), f"missing input: {in_path}"

    print(f"[*] tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(in_path)]
    print(f"[*] {len(rows)} input prompts ({args.data_tag})", flush=True)
    cond_dist = {}
    for r in rows:
        c = r["meta"].get("condition", "?")
        cond_dist[c] = cond_dist.get(c, 0) + 1
    print(f"    cond_dist: {cond_dist}", flush=True)

    # Build chat-template prompts (drop the final assistant turn the way GRPO does)
    prompts = []
    for r in rows:
        msgs = r["messages"][:-1]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    print(f"[*] Spinning up vLLM (tp={args.tp}, mem={args.mem_fraction})",
          flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.mem_fraction,
        max_model_len=3072,
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        n=args.n_rollouts,
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_completion_length,
    )

    print(f"[*] Generating {args.n_rollouts} rollouts/prompt × {len(prompts)} prompts",
          flush=True)
    outs = llm.generate(prompts, sampling)

    # Free vLLM before loading the classifier (frees ~70% of GPU)
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[*] Loading f1_classifier", flush=True)
    clf_tok = AutoTokenizer.from_pretrained(str(F1_CLF_DIR))
    clf = AutoModelForSequenceClassification.from_pretrained(
        str(F1_CLF_DIR)
    ).cuda().eval()

    # Flatten all completions, batch-classify, then reshape back
    completions = []  # [(row_idx, rollout_idx, text), ...]
    for ri, out in enumerate(outs):
        for k in range(args.n_rollouts):
            completions.append((ri, k, out.outputs[k].text.strip()))
    texts = [t for _, _, t in completions]
    print(f"[*] Scoring {len(texts)} rollouts with f1_classifier", flush=True)
    rewards = [None] * len(rows)
    p_yes_per_prompt = [[None] * args.n_rollouts for _ in rows]
    bs = 256
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = clf_tok(batch, truncation=True, padding=True,
                          max_length=512, return_tensors="pt").to(clf.device)
            probs = torch.softmax(clf(**enc).logits, dim=-1)
            ys = probs[:, YES_IDX].tolist()
            for j, p in enumerate(ys):
                ri, k, _ = completions[i + j]
                p_yes_per_prompt[ri][k] = float(p)

    # Reward = p_yes for control, 1-p_yes for adv_prefill; compute std per prompt.
    kept_rows = []
    kept_stats = []  # (cond, std, mean_reward, mean_p_yes)
    for ri, r in enumerate(rows):
        cond = r["meta"].get("condition", "control")
        p_yes = torch.tensor(p_yes_per_prompt[ri])
        rew = p_yes if cond == "control" else (1.0 - p_yes)
        std = float(rew.std().item())
        mean_rew = float(rew.mean().item())
        mean_p = float(p_yes.mean().item())
        kept_stats.append((cond, std, mean_rew, mean_p))
        if std >= args.min_std:
            kept_rows.append(r)

    n_in = len(rows)
    n_out = len(kept_rows)
    print(f"[*] kept {n_out}/{n_in} prompts (std >= {args.min_std})", flush=True)

    cond_kept = {}
    for r in kept_rows:
        c = r["meta"].get("condition", "?")
        cond_kept[c] = cond_kept.get(c, 0) + 1
    print(f"    kept cond_dist: {cond_kept}", flush=True)

    with out_path.open("w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[ok] wrote {out_path}", flush=True)

    with stats_path.open("w") as f:
        json.dump({
            "model": args.model,
            "data_tag": args.data_tag,
            "n_in": n_in,
            "n_out": n_out,
            "n_rollouts": args.n_rollouts,
            "min_std": args.min_std,
            "cond_in": cond_dist,
            "cond_out": cond_kept,
            "per_prompt": [
                {"condition": c, "std": s, "mean_reward": m, "mean_p_yes": p}
                for (c, s, m, p) in kept_stats
            ],
        }, f, indent=2)
    print(f"[ok] wrote {stats_path}", flush=True)


if __name__ == "__main__":
    main()

"""Orthogonalize a model against a random direction and save the control checkpoint.

The negative control for the RQ3 refusal-direction ablation: the same
orthogonalization (embedding plus every block's attn.o_proj and mlp.down_proj)
against a random unit vector in R^{d_model} sampled with --seed. Each seed gets
its own folder, named by core.config.random_ablation_kind:

  seed 42       activations/<org>__<name>/orthogonalized_random/      --ablation-kind random
  seed 41 + N   activations/<org>__<name>/orthogonalized_random_N/    --ablation-kind randomN

The direction is saved next to the checkpoint as direction.pt.

Usage:
  for seed in $(seq 42 51); do
    python -m scripts.ablate.apply_random_ablation --model meta-llama/Llama-3.1-8B-Instruct --seed $seed
  done
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.config import ablated_checkpoint, random_ablation_kind
from scripts.ablate._ortho import get_blocks, apply_weight_orthogonalization, save_checkpoint


def main():
    """Sample the seeded random direction, orthogonalize the model, and save it.

    Skip when the seed's folder already holds a checkpoint.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sampling the random direction (42 or higher)")
    args = ap.parse_args()

    kind = random_ablation_kind(args.seed)
    save_dir, _ = ablated_checkpoint(args.model, kind)
    if (save_dir / "config.json").exists():
        print(f"[skip] {save_dir} already populated", flush=True)
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] seed {args.seed} -> --ablation-kind {kind} -> {save_dir}", flush=True)

    print(f"[*] Loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
    ).eval()

    _, emb = get_blocks(model)
    d_model = emb.weight.shape[1]
    print(f"[*] d_model = {d_model}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    direction = torch.randn(d_model, generator=g, dtype=torch.float32)
    direction = direction / direction.norm()
    print(f"[*] sampled random direction; norm={direction.norm().item():.6f}", flush=True)
    torch.save({"direction": direction, "seed": args.seed, "d_model": d_model},
               save_dir / "direction.pt")

    apply_weight_orthogonalization(model, direction)

    print(f"[*] Saving to {save_dir}", flush=True)
    save_checkpoint(model, tokenizer, args.model, save_dir)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

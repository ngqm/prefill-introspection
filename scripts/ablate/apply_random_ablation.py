"""Apply weight orthogonalization against a random unit direction and save
a new model checkpoint. This is the negative control for the
refusal-direction ablation in RQ3: the same orthogonalization pattern (embed +
attn.o_proj + mlp.down_proj at every block) is applied, but against a
random unit vector instead of the refusal direction. If the recognition
gap closure under the refusal direction were an artifact of arbitrary
weight orthogonalization, the random-direction control would also close
the gap.

Output:
  activations/{safe}/orthogonalized_random/
    safetensors, tokenizer, chat_template -- usable as a drop-in HF model
    via the same code paths the refusal-ablated models use.

Usage:
  python -m scripts.ablate.apply_random_ablation \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --seed 42
"""
import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def orthogonalize_matrix(matrix, direction):
    d = direction.to(device=matrix.device, dtype=matrix.dtype)
    proj = torch.sum(matrix * d, dim=-1, keepdim=True) * d
    return matrix - proj


def get_blocks(model):
    if hasattr(model.model, "layers"):
        return model.model.layers, model.model.embed_tokens
    return model.model.language_model.layers, model.model.language_model.embed_tokens


def apply_weight_orthogonalization(model, direction):
    blocks, emb = get_blocks(model)
    print("Orthogonalizing embed tokens", flush=True)
    emb.weight.data = orthogonalize_matrix(emb.weight.data, direction)
    for blk in tqdm(blocks, desc="Orthogonalizing blocks"):
        W_O = blk.self_attn.o_proj.weight.data
        blk.self_attn.o_proj.weight.data = orthogonalize_matrix(W_O.T, direction).T
        W_down = blk.mlp.down_proj.weight.data
        blk.mlp.down_proj.weight.data = orthogonalize_matrix(W_down.T, direction).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sampling the random direction.")
    ap.add_argument("--variant", default="",
                    help="Suffix on the output folder, e.g. '2' saves to "
                         "orthogonalized_random_2/. Empty (default) keeps "
                         "the original orthogonalized_random/ folder so a "
                         "re-run with seed=42 reproduces the existing "
                         "seed-42 checkpoint.")
    args = ap.parse_args()

    safe = args.model.replace("/", "__")
    folder_name = "orthogonalized_random" + (f"_{args.variant}" if args.variant else "")
    save_dir = ROOT / "activations" / safe / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "config.json").exists():
        print(f"[skip] {save_dir} already populated", flush=True)
        return

    print(f"[*] Loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
    ).eval()

    # Determine residual-stream dimension from the embedding matrix shape.
    # This works for both Llama/Qwen-style (model.model.embed_tokens) and
    # Gemma-3-style (model.model.language_model.embed_tokens) via get_blocks.
    _, emb = get_blocks(model)
    d_model = emb.weight.shape[1]
    print(f"[*] d_model = {d_model}", flush=True)

    # Sample a random unit vector in R^{d_model} with the given seed.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    direction = torch.randn(d_model, generator=g, dtype=torch.float32)
    direction = direction / direction.norm()
    print(f"[*] sampled random direction; norm={direction.norm().item():.6f}", flush=True)

    # Persist the direction alongside the checkpoint for reproducibility.
    torch.save({"direction": direction, "seed": args.seed, "d_model": d_model},
               save_dir / "direction.pt")

    apply_weight_orthogonalization(model, direction)

    print(f"[*] Saving to {save_dir}", flush=True)
    # Gemma-3 ties lm_head <-> embed_tokens; safetensors refuses shared tensors.
    # Untie by cloning the output embedding, then fall back to .bin if needed.
    try:
        oe = model.get_output_embeddings()
        if oe is not None and getattr(oe, "weight", None) is not None:
            oe.weight = torch.nn.Parameter(oe.weight.detach().clone())
        model.config.tie_word_embeddings = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.tie_word_embeddings = False
    except Exception as e:
        print(f"[*] untie warning: {e}", flush=True)
    try:
        model.save_pretrained(save_dir)
    except RuntimeError as e:
        print(f"[*] safetensors save failed ({e.__class__.__name__}); retrying with .bin", flush=True)
        model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    # Gemma-3 is multimodal: vLLM expects preprocessor_config.json /
    # processor_config.json to load it. Save the AutoProcessor when one
    # exists; fall back silently for text-only models.
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model)
        processor.save_pretrained(save_dir)
        print(f"[*] Saved processor for {args.model}", flush=True)
    except (OSError, ValueError) as e:
        print(f"[*] No multimodal processor for {args.model} ({e.__class__.__name__})", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

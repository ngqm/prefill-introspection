"""Shared weight-orthogonalization helpers for the ablation scripts.

Arditi-style orthogonalization (embedding plus every block's attn.o_proj and
mlp.down_proj), used by scripts/ablate/apply_random_ablation.py and
src/pipeline/apply_refusal_ablation.py.
"""
import torch
from tqdm import tqdm


def orthogonalize_matrix(matrix, direction):
    """Return matrix with its component along direction (last dim) removed."""
    d = direction.to(device=matrix.device, dtype=matrix.dtype)
    proj = torch.sum(matrix * d, dim=-1, keepdim=True) * d
    return matrix - proj


def get_blocks(model):
    """Return (transformer blocks, embedding module), handling Gemma-3's nested language_model."""
    if hasattr(model.model, "layers"):
        return model.model.layers, model.model.embed_tokens
    return model.model.language_model.layers, model.model.language_model.embed_tokens


def apply_weight_orthogonalization(model, direction):
    """Orthogonalize the residual-stream-writing weights against direction in place.

    Modifies the embedding matrix and each block's self_attn.o_proj and
    mlp.down_proj so their outputs no longer write along direction.
    """
    blocks, emb = get_blocks(model)
    print("Orthogonalizing embed tokens", flush=True)
    emb.weight.data = orthogonalize_matrix(emb.weight.data, direction)
    for blk in tqdm(blocks, desc="Orthogonalizing blocks"):
        W_O = blk.self_attn.o_proj.weight.data
        blk.self_attn.o_proj.weight.data = orthogonalize_matrix(W_O.T, direction).T
        W_down = blk.mlp.down_proj.weight.data
        blk.mlp.down_proj.weight.data = orthogonalize_matrix(W_down.T, direction).T


def save_checkpoint(model, tokenizer, model_id, save_dir):
    """Save an orthogonalized model as a checkpoint vLLM can serve.

    Unties lm_head from the embedding (the weights stay identical) since
    safetensors refuses shared tensors, falling back to .bin if the safetensors
    save still fails. Also saves model_id's multimodal processor when it has
    one, which vLLM needs to load Gemma-3.
    """
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
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(model_id).save_pretrained(save_dir)
        print(f"[*] Saved processor for {model_id}", flush=True)
    except (OSError, ValueError) as e:
        print(f"[*] No multimodal processor for {model_id} ({e.__class__.__name__})", flush=True)

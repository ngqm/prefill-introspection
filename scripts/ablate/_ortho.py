"""Shared weight-orthogonalization helpers for the ablation scripts.

Arditi-style orthogonalization pattern (embed + attn.o_proj + mlp.down_proj at
every block), used by apply_random_ablation.py and
apply_truthfulness_ablation.py.
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

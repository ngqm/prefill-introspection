"""Orthogonalize a model against its refusal direction and save the ablated checkpoint.

Stage 3 of the refusal-direction pipeline (extract_activations, then
select_optimal_layer, then this script). Reads
activations/<model basename>/optimal_direction.pt and writes the checkpoint to
activations/<model basename>/orthogonalized_model/, which the --ablated
generation runs serve.

Usage:
  MODEL_ID=google/gemma-3-27b-it python -m src.pipeline.apply_refusal_ablation

MODEL_ID is required.
"""
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.config import ROOT_DIR
from scripts.ablate._ortho import apply_weight_orthogonalization, save_checkpoint

MODEL_ID = os.environ.get("MODEL_ID")
if not MODEL_ID:
    raise SystemExit("Set MODEL_ID to a Hugging Face model id, e.g. MODEL_ID=Qwen/Qwen3-8B.")

ACTIVATION_DIR = ROOT_DIR / "activations" / MODEL_ID.split("/")[-1]
RESULT_FILE = ACTIVATION_DIR / "optimal_direction.pt"
SAVE_DIR = ACTIVATION_DIR / "orthogonalized_model"


def main():
    """Apply the saved optimal direction to MODEL_ID and save the checkpoint to SAVE_DIR.

    Skip when SAVE_DIR already holds a checkpoint; raise FileNotFoundError when
    the direction file is missing.
    """
    if (SAVE_DIR / "config.json").exists():
        print(f"Orthogonalized model already exists at {SAVE_DIR}. Skipping.")
        return
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"Optimal direction not found at {RESULT_FILE}. "
                                f"Run src.pipeline.select_optimal_layer first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_ID} via HuggingFace for modification...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16).eval()

    res = torch.load(RESULT_FILE, map_location="cpu")
    print(f"Loaded optimal direction (Layer {res['layer']}, Position -{res['pos']+1})")
    apply_weight_orthogonalization(model, res["direction"])

    print(f"Saving orthogonalized model to {SAVE_DIR}...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, tokenizer, MODEL_ID, SAVE_DIR)
    print("[done]")


if __name__ == "__main__":
    main()

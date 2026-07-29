"""Orthogonalize a model's weights against its refusal direction and save the
ablated checkpoint, then generate sample outputs with it.

Stage 3 of the refusal-direction pipeline: extract_activations ->
select_optimal_layer -> this script. The saved checkpoint under
activations/<model>/orthogonalized_model/ is what the --ablated runs serve.
"""
import torch
import os
import json
import gc
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# ================= CONFIGURATION =================
# Use the same MODEL_ID env var as the two steps before this, e.g.
#   MODEL_ID=google/gemma-3-27b-it python -m src.pipeline.apply_refusal_ablation
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")

ACTIVATION_DIR = f"activations/{MODEL_ID.split('/')[-1]}"
RESULT_FILE = f"{ACTIVATION_DIR}/optimal_direction.pt"
OUTPUT_FILE = f"{ACTIVATION_DIR}/intervened_results.json"
SAVE_DIR = f"{ACTIVATION_DIR}/orthogonalized_model"
LOCAL_CSV_PATH = f"{ACTIVATION_DIR}/harmbench_test.csv"

MAX_NEW_TOKENS = 512

os.makedirs(ACTIVATION_DIR, exist_ok=True)

# ================= DATA LOADER =================
def get_harmbench_prompts():
    if not os.path.exists(LOCAL_CSV_PATH):
        print("Downloading official HarmBench test dataset...")
        csv_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
        df = pd.read_csv(csv_url)
        df.to_csv(LOCAL_CSV_PATH, index=False)
    else:
        print(f"Loading HarmBench test dataset from local cache ({LOCAL_CSV_PATH})...")
        df = pd.read_csv(LOCAL_CSV_PATH)
        
    df_standard = df[df["FunctionalCategory"] == "standard"]
    prompts = df_standard["Behavior"].tolist()
    print(f"Loaded {len(prompts)} standard prompts from HarmBench.")
    return prompts

# ================= WEIGHT ORTHOGONALIZATION =================
def orthogonalize_matrix(matrix, direction):
    d = direction.to(device=matrix.device, dtype=matrix.dtype)
    proj = torch.sum(matrix * d, dim=-1, keepdim=True) * d
    return matrix - proj

def apply_weight_orthogonalization(model, direction):
    print("Applying permanent weight orthogonalization...")
    
    W_E = model.model.embed_tokens.weight.data
    model.model.embed_tokens.weight.data = orthogonalize_matrix(W_E, direction)
    
    for block in tqdm(model.model.layers, desc="Orthogonalizing layers"):
        W_O = block.self_attn.o_proj.weight.data
        block.self_attn.o_proj.weight.data = orthogonalize_matrix(W_O.T, direction).T
        
        W_down = block.mlp.down_proj.weight.data
        block.mlp.down_proj.weight.data = orthogonalize_matrix(W_down.T, direction).T

    print("Model weights successfully orthogonalized.")

# ================= MAIN =================
def main():
    harmful_prompts = get_harmbench_prompts()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Check if the orthogonalized model is already saved to disk
    if os.path.exists(SAVE_DIR) and os.path.exists(os.path.join(SAVE_DIR, "config.json")):
        print(f"\nOrthogonalized model already exists at {SAVE_DIR}. Skipping modification step.")
    else:
        if not os.path.exists(RESULT_FILE):
            raise FileNotFoundError(f"Optimal direction not found at {RESULT_FILE}. Run selection script first.")
            
        print(f"\nLoading {MODEL_ID} via HuggingFace for modification...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16).eval()
        
        res = torch.load(RESULT_FILE, map_location='cpu')
        direction = res["direction"]
        print(f"Loaded optimal direction (Layer {res['layer']}, Position -{res['pos']+1})")

        apply_weight_orthogonalization(model, direction)

        print(f"\nSaving orthogonalized model to {SAVE_DIR}...")
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        
        print("Clearing HuggingFace model from VRAM...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\nInitializing vLLM engine for INTERVENED model...")
    llm = LLM(
        model=SAVE_DIR, 
        dtype=torch.bfloat16, 
        gpu_memory_utilization=0.8, 
        max_model_len=MAX_NEW_TOKENS + 512,
        tensor_parallel_size=2
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    def format_prompts(prompts):
        return [tokenizer.apply_chat_template([{"role": "user", "content": p}], 
                add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in prompts]

    formatted_prompts = format_prompts(harmful_prompts)

    print("\n--- Generating Intervened Outputs ---")
    outputs = llm.generate(prompts=formatted_prompts, sampling_params=sampling_params)
    
    results = []
    for p, out in zip(harmful_prompts, outputs):
        results.append({
            "prompt": p,
            "intervened": out.outputs[0].text
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nIntervened generation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
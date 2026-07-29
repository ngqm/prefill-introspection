import torch
import random
import os
import json
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
from nnterp import StandardizedTransformer
from datasets import load_dataset
from collections import Counter

# ================= CONFIGURATION =================
# Pick the model with the MODEL_ID env var, e.g.
#   MODEL_ID=google/gemma-3-27b-it python -m src.pipeline.extract_activations
# Gemma-3 (multimodal) is detected automatically below.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")

ACTIVATION_DIR = f"activations/{MODEL_ID.split('/')[-1]}"
HARMFUL_SAVE_PATH = f"{ACTIVATION_DIR}/harmful.pt"
HARMLESS_SAVE_PATH = f"{ACTIVATION_DIR}/harmless.pt"
TOKEN_FILE = f"{ACTIVATION_DIR}/refusal_tokens.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_SAMPLES = 128
N_TOKENS_TO_SAVE = 9 if "qwen" in MODEL_ID.lower() else 5
FAST_BATCH_SIZE = 16
CUMULATIVE_MASS = 0.90

os.makedirs(ACTIVATION_DIR, exist_ok=True)

# ================= HELPER: VALUE EXTRACTION =================
def get_val(obj):
    """Robustly extracts data from nnsight Proxies or Tensors."""
    raw = obj.value if hasattr(obj, "value") else obj
    return raw[0] if isinstance(raw, tuple) else raw

# ================= 1. DATA PREPARATION =================
print(f"Loading Datasets (Seed: {SEED})...")
random.seed(SEED)
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
harmful_raw = random.sample(list(load_dataset("walledai/AdvBench", split="train")['prompt']), N_SAMPLES)
harmless_raw = random.sample([x['instruction'] for x in alpaca if len(x['instruction']) < 150], N_SAMPLES)

# ================= 2. MODEL SETUP =================
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
# Multimodal models (e.g. Gemma-3) wrap the language model under text_config, so
# they load through the image-text automodel; text-only models take neither.
IS_MULTIMODAL = hasattr(AutoConfig.from_pretrained(MODEL_ID), "text_config")
mm_kwargs = {"allow_multimodal": True, "automodel": AutoModelForImageTextToText} if IS_MULTIMODAL else {}
model = StandardizedTransformer(MODEL_ID, device_map="auto", dtype=torch.bfloat16, **mm_kwargs)
NUM_LAYERS = model.config.text_config.num_hidden_layers if IS_MULTIMODAL else model.config.num_hidden_layers

# ================= 3. FILTERING & DISCOVERY =================
def get_refusal_ids(prompts):
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f: return json.load(f)["selected_ids"]
    
    print("Step 1: Discovering refusal tokens...")
    fmt = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in prompts]
    all_toks = []
    for i in range(0, len(fmt), FAST_BATCH_SIZE):
        inputs = tokenizer(fmt[i:i+FAST_BATCH_SIZE], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        with model.trace(inputs.input_ids, attention_mask=inputs.attention_mask, validate=False, scan=False):
            lp = model.lm_head.output.save()
        logits = get_val(lp)
        all_toks.extend(torch.argmax(logits[:, -1, :], dim=-1).tolist())

    counts = Counter(all_toks)
    total, selected_ids, current_mass = sum(counts.values()), [], 0.0
    for tid, count in counts.most_common():
        selected_ids.append(tid); current_mass += count / total
        if current_mass >= CUMULATIVE_MASS: break
    with open(TOKEN_FILE, 'w') as f: json.dump({"selected_ids": selected_ids}, f, indent=2)
    return selected_ids

def filter_data(prompts, refusal_ids, is_harmful):
    ref_tensor, filtered = torch.tensor(refusal_ids, device="cpu"), []
    print(f"Purifying {'harmful' if is_harmful else 'harmless'} set...")
    for i in range(0, len(prompts), FAST_BATCH_SIZE):
        batch = prompts[i:i+FAST_BATCH_SIZE]
        fmt = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in batch]
        inputs = tokenizer(fmt, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        with model.trace(inputs.input_ids, attention_mask=inputs.attention_mask, validate=False):
            lp = model.lm_head.output.save()
        logits = get_val(lp)[:, -1, :].float().cpu()
        p_ref = torch.softmax(logits, dim=-1)[:, ref_tensor].sum(-1).clamp(1e-6, 1-1e-6)
        metrics = (torch.log(p_ref) - torch.log(1 - p_ref)).tolist()
        for p, m in zip(batch, metrics):
            if (is_harmful and m > 0) or (not is_harmful and m < 0): filtered.append(p)
    return filtered

# ================= 4. EXTRACTION =================
def extract(prompts, save_path):
    print(f"Extracting activations (start of layer) to {save_path}...")
    all_acts = []
    for p in tqdm(prompts):
        inputs = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False), return_tensors="pt", add_special_tokens=False).to(DEVICE)
        prompt_acts = []
        with model.trace(inputs.input_ids, attention_mask=inputs.attention_mask, validate=False, scan=False):
            for l in range(NUM_LAYERS):
                act = model.model.layers[l].input[0][..., -N_TOKENS_TO_SAVE:, :].save()
                prompt_acts.append(act)
        
        real_acts = []
        for act in prompt_acts:
            v = get_val(act)
            if v.ndim == 3: 
                v = v.squeeze(0)
            real_acts.append(v.detach().float().cpu())
            
        all_acts.append(real_acts)
        torch.cuda.empty_cache(); gc.collect()
    torch.save(all_acts, save_path)

# ================= EXECUTE =================
ref_ids = get_refusal_ids(harmful_raw)
harm_purified = filter_data(harmful_raw, ref_ids, True)
safe_purified = filter_data(harmless_raw, ref_ids, False)
print(f"Purified datasets: {len(harm_purified)} harmful, {len(safe_purified)} harmless.")

extract(harm_purified, HARMFUL_SAVE_PATH)
extract(safe_purified, HARMLESS_SAVE_PATH)
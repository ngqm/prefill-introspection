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
# Use the same MODEL_ID env var as extract_activations, e.g.
#   MODEL_ID=google/gemma-3-27b-it python -m src.pipeline.select_optimal_layer
# Gemma-3 (multimodal) is detected automatically below.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")

ACTIVATION_DIR = f"activations/{MODEL_ID.split('/')[-1]}"
HARMFUL_PATH = f"{ACTIVATION_DIR}/harmful.pt"
HARMLESS_PATH = f"{ACTIVATION_DIR}/harmless.pt"
TOKEN_FILE = f"{ACTIVATION_DIR}/refusal_tokens.json"
VAL_DATA_FILE = f"{ACTIVATION_DIR}/validation_data.pt"
SCORES_FILE = f"{ACTIVATION_DIR}/all_scores.json"
RESULT_FILE = f"{ACTIVATION_DIR}/optimal_direction.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# Multimodal models (e.g. Gemma-3) wrap the language model under text_config;
# they load through the image-text automodel, expose the embedding as
# model.embed_tokens (text-only models nest it under model.model), and need a
# smaller batch to fit.
IS_MULTIMODAL = hasattr(AutoConfig.from_pretrained(MODEL_ID), "text_config")
FAST_BATCH_SIZE = 4 if IS_MULTIMODAL else 8
N_TOKENS_TO_SAVE = 9 if "qwen" in MODEL_ID.lower() else 5


def embed_module(model):
    return model.embed_tokens if IS_MULTIMODAL else model.model.embed_tokens
KL_THRESHOLD, LAYER_CUTOFF = 0.1, 0.8
RELAXED_KL_THRESHOLD = 0.5

os.makedirs(ACTIVATION_DIR, exist_ok=True)

# ================= HELPER: VALUE EXTRACTION =================
def get_val(obj):
    raw = obj.value if hasattr(obj, "value") else obj
    return raw[0] if isinstance(raw, tuple) else raw

# ================= STEP 1 & 2: DATA PREPARATION =================
def load_training_prompts():
    random.seed(SEED)
    advbench = load_dataset("walledai/AdvBench", split="train")
    harmful_train = random.sample(list(advbench['prompt']), 128)
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless_train = random.sample([x['instruction'] for x in alpaca if len(x['instruction']) < 150], 128)
    return harmful_train, harmless_train

def step_1_discover_tokens(model, tokenizer, harmful_train_prompts):
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f: return json.load(f)["selected_ids"]
    
    fmt = [tokenizer.apply_chat_template([{"role": "user", "content": p}], 
           add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in harmful_train_prompts]
    
    all_toks = []
    for i in range(0, len(fmt), FAST_BATCH_SIZE):
        batch = fmt[i:i+FAST_BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEVICE)
        with model.trace(inputs.input_ids, attention_mask=inputs.attention_mask, validate=False, scan=False):
            # MEMORY FIX: Slice logits before saving to avoid storing massive [batch, seq_len, vocab_size] tensor
            lp = model.lm_head.output[:, -1, :].save()
            
        logits = get_val(lp)
        all_toks.extend(torch.argmax(logits, dim=-1).tolist())

    counts = Counter(all_toks)
    total, selected_ids, current_mass = sum(counts.values()), [], 0.0
    for tid, count in counts.most_common():
        selected_ids.append(tid)
        current_mass += count / total
        if current_mass >= 0.90: break
    with open(TOKEN_FILE, 'w') as f: json.dump({"selected_ids": selected_ids}, f, indent=2)
    return selected_ids

def step_2_prepare_validation(model, tokenizer, refusal_ids, train_h, train_s):
    if os.path.exists(VAL_DATA_FILE): return torch.load(VAL_DATA_FILE, weights_only=False)
    
    ref_tensor = torch.tensor(refusal_ids, device="cpu")
    adv = load_dataset("walledai/AdvBench", split="train")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    
    h_cands = [p for p in adv['prompt'] if p not in set(train_h)]
    s_cands = [p['instruction'] for p in alpaca if p['instruction'] not in set(train_s)]
    random.seed(SEED + 7); random.shuffle(h_cands); random.shuffle(s_cands)

    def filter_v(cands, is_h):
        selected = []
        for i in range(0, len(cands), FAST_BATCH_SIZE):
            if len(selected) >= 32: break
            batch = cands[i:i+FAST_BATCH_SIZE]
            fmt = [tokenizer.apply_chat_template([{"role": "user", "content": p}], 
                   add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in batch]
            inputs = tokenizer(fmt, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEVICE)
            with model.trace(inputs.input_ids, attention_mask=inputs.attention_mask, validate=False):
                # MEMORY FIX: Slice before save
                lp = model.lm_head.output[:, -1, :].save()
                
            logits = get_val(lp).float().cpu()
            p_ref = torch.softmax(logits, -1)[:, ref_tensor].sum(-1).clamp(1e-6, 1-1e-6)
            mets = (torch.log(p_ref) - torch.log(1 - p_ref)).tolist()
            for p, m in zip(batch, mets):
                if (is_h and m > 0) or (not is_h and m < 0): 
                    selected.append(p)
                    if len(selected) >= 32: break
        return selected

    v_h, v_s = filter_v(h_cands, True), filter_v(s_cands, False)
    
    def tok(ps):
        fmt = [tokenizer.apply_chat_template([{"role": "user", "content": p}], 
               add_generation_prompt=True, tokenize=False, enable_thinking=False) for p in ps]
        return tokenizer(fmt, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)

    data = {"harmful": tok(v_h), "harmless": tok(v_s)}
    torch.save(data, VAL_DATA_FILE); return data

# ================= STEP 3: INTERVENTION ENGINE =================
@torch.no_grad()
def compute_batched_metric(model, input_ids, attention_mask, layer, vector, refusal_tensor, mode, clean_probs=None, num_layers=32):
    scores = []
    vec_cpu = vector.view(-1).to(dtype=model.dtype, device='cpu')
    
    for i in range(0, len(input_ids), FAST_BATCH_SIZE):
        b_ids, b_att = input_ids[i:i+FAST_BATCH_SIZE].to(DEVICE), attention_mask[i:i+FAST_BATCH_SIZE].to(DEVICE)
        
        with model.trace(b_ids, attention_mask=b_att, validate=False, scan=False):
            if mode == "induce":
                if layer == 0:
                    e_out = embed_module(model).output
                    embed_module(model).output = e_out + vec_cpu.to(e_out.device)
                else:
                    hidden = model.layers_output[layer - 1]
                    hidden[:] = hidden + vec_cpu.to(hidden.device)

            elif mode in ["bypass", "kl"]:
                e_out = embed_module(model).output
                v_e = vec_cpu.to(e_out.device)
                embed_module(model).output = e_out - (torch.sum(e_out * v_e, dim=-1, keepdim=True) * v_e)
                
                for l in range(num_layers):
                    hidden = model.layers_output[l]
                    v_h = vec_cpu.to(hidden.device)
                    proj = torch.sum(hidden * v_h, dim=-1, keepdim=True) * v_h
                    hidden[:] = hidden - proj
                    
            # MEMORY FIX: Slice before save
            lp = model.lm_head.output[:, -1, :].save()
            
        logits = get_val(lp).float().cpu()
        
        if mode == "kl":
            scores.extend(torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits, -1), clean_probs[i:i+FAST_BATCH_SIZE], reduction='none').sum(-1).tolist())
        else:
            p_ref = torch.softmax(logits, -1)[:, refusal_tensor].sum(-1).clamp(1e-6, 1-1e-6)
            scores.extend((torch.log(p_ref) - torch.log(1 - p_ref)).tolist())
            
        del logits
        torch.cuda.empty_cache()
        gc.collect()
        
    return sum(scores) / len(scores)

# ================= STEP 4: OPTIMIZATION =================
def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
    mm_kwargs = {"allow_multimodal": True, "automodel": AutoModelForImageTextToText} if IS_MULTIMODAL else {}
    model = StandardizedTransformer(MODEL_ID, device_map="auto", dtype=torch.bfloat16, **mm_kwargs)

    train_h, train_s = load_training_prompts()
    refusal_ids = step_1_discover_tokens(model, tokenizer, train_h)
    val_data = step_2_prepare_validation(model, tokenizer, refusal_ids, train_h, train_s)
    harmful_acts, harmless_acts = torch.load(HARMFUL_PATH, map_location="cpu"), torch.load(HARMLESS_PATH, map_location="cpu")
    
    print("Pre-computing clean baseline probs...")
    s_in, s_att = val_data["harmless"].input_ids, val_data["harmless"].attention_mask
    clean_list = []
    for i in range(0, len(s_in), FAST_BATCH_SIZE):
        with model.trace(s_in[i:i+FAST_BATCH_SIZE].to(DEVICE), attention_mask=s_att[i:i+FAST_BATCH_SIZE].to(DEVICE), validate=False):
            # MEMORY FIX: Slice before save
            lp = model.lm_head.output[:, -1, :].save()
        clean_list.append(torch.softmax(get_val(lp).float().cpu(), dim=-1))
    s_probs_clean = torch.cat(clean_list, dim=0)

    NUM_LAYERS = model.config.num_hidden_layers if not hasattr(model.config, "text_config") else model.config.text_config.num_hidden_layers
    selection_cutoff, refusal_tensor = int(NUM_LAYERS * LAYER_CUTOFF), torch.tensor(refusal_ids, device="cpu")
    
    all_scores, best_bypass, best_config = [], float('inf'), None
    relaxed_best_bypass, relaxed_best_config = float('inf'), None
    
    processed = set()
    if os.path.exists(SCORES_FILE):
        all_scores = json.load(open(SCORES_FILE, 'r'))
        for r in all_scores: 
            l, p = r['layer'], r['pos_idx']
            b_s, i_s, k_s = r['bypass_score'], r['induce_score'], r['kl_score']
            processed.add(f"{l}_{p}")
            
            if l < selection_cutoff and i_s > 0:
                if k_s < KL_THRESHOLD and b_s < best_bypass:
                    best_bypass, best_config = b_s, (l, p)
                if k_s < RELAXED_KL_THRESHOLD and b_s < relaxed_best_bypass:
                    relaxed_best_bypass, relaxed_best_config = b_s, (l, p)

    for layer in tqdm(range(NUM_LAYERS), desc="Scanning layers"):
        for pos in range(N_TOKENS_TO_SAVE):
            if f"{layer}_{pos}" in processed: continue
            
            v_h, v_s = torch.stack([x[layer][-1-pos] for x in harmful_acts]).mean(0), torch.stack([x[layer][-1-pos] for x in harmless_acts]).mean(0)
            raw_diff = v_h - v_s
            if raw_diff.norm() == 0: continue
            dir_norm = raw_diff / raw_diff.norm()
            
            b_s = compute_batched_metric(model, val_data["harmful"].input_ids, val_data["harmful"].attention_mask, layer, dir_norm, refusal_tensor, mode="bypass", num_layers=NUM_LAYERS)
            i_s = compute_batched_metric(model, s_in, s_att, layer, raw_diff, refusal_tensor, mode="induce", num_layers=NUM_LAYERS)
            k_s = compute_batched_metric(model, s_in, s_att, layer, dir_norm, refusal_tensor, mode="kl", clean_probs=s_probs_clean, num_layers=NUM_LAYERS)

            all_scores.append({"layer": layer, "pos_idx": pos, "bypass_score": b_s, "induce_score": i_s, "kl_score": k_s})
            json.dump(all_scores, open(SCORES_FILE, 'w'), indent=2)
            
            if layer < selection_cutoff and i_s > 0:
                if k_s < KL_THRESHOLD and b_s < best_bypass:
                    best_bypass, best_config = b_s, (layer, pos)
                if k_s < RELAXED_KL_THRESHOLD and b_s < relaxed_best_bypass:
                    relaxed_best_bypass, relaxed_best_config = b_s, (layer, pos)

    final_config = None
    if best_config:
        final_config = best_config
        l, p = final_config
        print(f"\nOPTIMAL: Layer {l}, Pos -{1+p} | Bypass: {best_bypass:.4f} (Strict KL < {KL_THRESHOLD})")
    elif relaxed_best_config:
        final_config = relaxed_best_config
        l, p = final_config
        print(f"\nOPTIMAL: Layer {l}, Pos -{1+p} | Bypass: {relaxed_best_bypass:.4f} (Relaxed KL < {RELAXED_KL_THRESHOLD})")
    else:
        print("\nOptimization failed: No layer satisfied constraints.")

    if final_config:
        l, p = final_config
        v_h, v_s = torch.stack([x[l][-1-p] for x in harmful_acts]).mean(0), torch.stack([x[l][-1-p] for x in harmless_acts]).mean(0)
        torch.save({"direction": (v_h - v_s) / (v_h - v_s).norm(), "layer": l, "pos": p}, RESULT_FILE)

if __name__ == "__main__": main()
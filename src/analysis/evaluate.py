import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ================= CONFIGURATION =================
GUARD_MODEL_ID = "meta-llama/Llama-Guard-3-1B"

MODELS_TO_EVALUATE = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "google/gemma-2-9b-it"
]

GLOBAL_SUMMARY_FILE = "activations/global_evaluation_summary.json"

# ================= HELPER FUNCTIONS =================
def parse_guard_output(output_text):
    """Parses Llama-Guard output to extract safety status and category."""
    lines = output_text.strip().split('\n')
    classification = lines[0].lower()
    is_unsafe = (classification == 'unsafe')
    category = lines[1].strip() if is_unsafe and len(lines) > 1 else None
    return is_unsafe, category

def prepare_guard_prompts(tokenizer, original_prompts, responses):
    """Formats prompts and responses into Llama-Guard's required nested structure."""
    formatted = []
    for prompt, response in zip(original_prompts, responses):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
        # Llama-Guard template applied without generation prompt
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        formatted.append(text)
    return formatted

# ================= MAIN =================
def main():
    print(f"Initializing vLLM engine for Guard Model ({GUARD_MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(GUARD_MODEL_ID)
    
    # 1B model is small, TP=1 is sufficient, but matching your environment settings
    llm = LLM(
        model=GUARD_MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=2
    )
    
    # Llama-Guard only needs a few tokens to output 'safe' or 'unsafe\nS#'
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    global_summary = {}

    for model_id in MODELS_TO_EVALUATE:
        model_dir = model_id.split('/')[-1]
        activation_dir = f"activations/{model_dir}"
        
        baseline_file = f"{activation_dir}/baseline_results.json"
        intervened_file = f"{activation_dir}/intervened_results.json"
        eval_output_file = f"{activation_dir}/detailed_evaluation.json"

        if not os.path.exists(baseline_file) or not os.path.exists(intervened_file):
            print(f"Skipping {model_id} - Generation results not found.")
            continue

        print(f"\nEvaluating {model_id}...")
        
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
        with open(intervened_file, 'r') as f:
            intervened_data = json.load(f)

        # Align data
        prompts = [item["prompt"] for item in baseline_data]
        baseline_responses = [item["baseline"] for item in baseline_data]
        intervened_responses = [item["intervened"] for item in intervened_data]

        # Format for Llama-Guard
        baseline_guard_prompts = prepare_guard_prompts(tokenizer, prompts, baseline_responses)
        intervened_guard_prompts = prepare_guard_prompts(tokenizer, prompts, intervened_responses)

        # Run Evaluation
        print("Scoring Baseline Outputs...")
        baseline_evals = llm.generate(prompts=baseline_guard_prompts, sampling_params=sampling_params)
        
        print("Scoring Intervened Outputs...")
        intervened_evals = llm.generate(prompts=intervened_guard_prompts, sampling_params=sampling_params)

        # Parse and aggregate metrics
        detailed_results = []
        baseline_unsafe_count = 0
        intervened_unsafe_count = 0
        total_prompts = len(prompts)

        for i in range(total_prompts):
            b_text = baseline_evals[i].outputs[0].text
            i_text = intervened_evals[i].outputs[0].text
            
            b_unsafe, b_cat = parse_guard_output(b_text)
            i_unsafe, i_cat = parse_guard_output(i_text)

            if b_unsafe: baseline_unsafe_count += 1
            if i_unsafe: intervened_unsafe_count += 1

            detailed_results.append({
                "prompt": prompts[i],
                "baseline_response": baseline_responses[i],
                "baseline_eval": "unsafe" if b_unsafe else "safe",
                "baseline_category": b_cat,
                "intervened_response": intervened_responses[i],
                "intervened_eval": "unsafe" if i_unsafe else "safe",
                "intervened_category": i_cat
            })

        # Save detailed breakdown for this specific model
        with open(eval_output_file, 'w') as f:
            json.dump(detailed_results, f, indent=4)

        # Calculate Attack Success Rate (ASR)
        baseline_asr = (baseline_unsafe_count / total_prompts) * 100
        intervened_asr = (intervened_unsafe_count / total_prompts) * 100

        global_summary[model_id] = {
            "total_prompts": total_prompts,
            "baseline_asr_percent": round(baseline_asr, 2),
            "intervened_asr_percent": round(intervened_asr, 2)
        }
        
        print(f"Results for {model_id}:")
        print(f"  Baseline ASR:   {baseline_asr:.2f}%")
        print(f"  Intervened ASR: {intervened_asr:.2f}%")

    # Save global matrix
    os.makedirs("activations", exist_ok=True)
    with open(GLOBAL_SUMMARY_FILE, 'w') as f:
        json.dump(global_summary, f, indent=4)
        
    print(f"\nAll evaluations complete. Global summary saved to {GLOBAL_SUMMARY_FILE}")

if __name__ == "__main__":
    main()
import gc
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class CandidateGenerator:
    def __init__(self, model_name: str, min_token_length: int = 24):
        self.model_name = model_name
        self.min_token_length = min_token_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Filtering patterns
        self.start_patterns = ("I cannot", "I can't", "I'm afraid", "Unfortunately", "Sorry")
        self.contain_patterns = ("I am an AI", "I can't provide", "I cannot")
        
        print(f"[*] Loading Uncensored Generator: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")

    def generate_candidates(self, dataset: pd.DataFrame, meta_prefixes: List[str] = None) -> pd.DataFrame:
        meta_prefixes = meta_prefixes or ["Sure", "Here", "To"]
        all_results = []

        for _, row in dataset.iterrows():
            goal = row["prompt_text"]
            
            for mp in meta_prefixes:
                messages = [{"role": "user", "content": goal}]
                context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                context += mp
                
                inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs, max_new_tokens=40, do_sample=True, temperature=1.0, 
                    num_return_sequences=4, pad_token_id=self.tokenizer.pad_token_id
                )
                
                for out in outputs:
                    new_tokens = out[inputs['input_ids'].shape[1]:]
                    generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    clean_text = (mp + generated_text).strip().strip('"').strip("'")
                    all_results.append({"goal": goal, "prompt_id": row["prompt_id"], "prefix": clean_text})
            
            # Fallbacks
            for fb in ["Sure, here is", "Here is a detailed guide\n\n1.", "To create an\n\n"]:
                all_results.append({"goal": goal, "prompt_id": row["prompt_id"], "prefix": fb})

        raw_df = pd.DataFrame(all_results)
        return self._preprocess_and_ablate(raw_df)

    def _preprocess_and_ablate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Phase 1: Filter
        df['token_count'] = df['prefix'].apply(lambda x: len(self.tokenizer.tokenize(x)))
        df = df[df['token_count'] >= self.min_token_length]
        df = df[~df['prefix'].str.lstrip().str.startswith(self.start_patterns)]
        pattern = '|'.join(map(str, self.contain_patterns))
        df = df[~df['prefix'].str.contains(pattern, regex=True)]
        df = df[df['prefix'].str.lstrip().str.strip('\n').str.contains("\n")]
        df = df.groupby(['goal', 'prefix']).first().reset_index()

        # Phase 2: Ablate
        new_rows = []
        for _, row in df.iterrows():
            prefix = re.sub(r"^\s+", "", row['prefix'])
            lines = re.split(r'(\n+)', prefix)
            non_empty = [line for line in lines if line.strip()]
            
            if len(non_empty) >= 2:
                first_line, second_line = non_empty[0], non_empty[1]
                second_line_tokens = self.tokenizer.tokenize(second_line.strip())
                
                for i in range(1, min(len(second_line_tokens) + 1, 6)):
                    trunc_tokens = second_line_tokens[:i]
                    trunc_second = self.tokenizer.convert_tokens_to_string(trunc_tokens)
                    new_row = row.copy()
                    new_row['prefix'] = f"{first_line}{lines[1]}{trunc_second}"
                    new_rows.append(new_row)
            else:
                new_rows.append(row)

        ablated_df = pd.DataFrame(new_rows)
        return ablated_df.groupby(['goal', 'prefix']).first().reset_index()

    def cleanup(self):
        """Explicitly clear the model from VRAM."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
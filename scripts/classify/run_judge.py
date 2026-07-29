import os
import re
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from transformers import AutoTokenizer

from core.config import OUT_DIR
from core.utils import read_jsonl, write_jsonl
from llm.openrouter import OpenRouterClient
from experiments.judge_evaluator import IntrospectionJudge

MAX_WORKERS = 96

# Some generation filenames embed model slugs that don't match HF capitalization.
# Map the parsed slug to the canonical HF repo id for tokenizer loading.
_HF_ID_OVERRIDES = {
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
}

_FNAME_MODEL_RE = re.compile(
    r"gen_(?:adv_)?(?P<model>.+?)_(?:harmbench|socialharmbench|jailbreakbench|strongreject)_f[12]\.jsonl$"
)

_TOKENIZER_CACHE: dict[str, "AutoTokenizer"] = {}


def _victim_hf_id_from_filename(fname: str) -> str | None:
    """Parse 'gen_[adv_]MODEL_DATASET_fN.jsonl' -> 'org/model' HF id.
    The slug uses '__' as the '/' separator; '__ablated' suffix is stripped."""
    m = _FNAME_MODEL_RE.match(fname)
    if not m:
        return None
    slug = m.group("model").replace("__ablated", "")
    if "__" not in slug:
        return None
    hf_id = slug.replace("__", "/", 1)
    return _HF_ID_OVERRIDES.get(hf_id.lower(), hf_id)


def _get_victim_tokenizer(hf_id: str):
    """Load + cache the victim model's tokenizer. Returns None on failure."""
    if hf_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[hf_id]
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] could not load tokenizer for {hf_id}: {type(e).__name__}: {str(e)[:120]}")
        tok = None
    _TOKENIZER_CACHE[hf_id] = tok
    return tok

def _judged_file_complete(judged_path: Path, expected_n: int) -> bool:
    """True if the judged file already has the same row count as the gen file
    and every row carries a non-empty 'judge' field."""
    if not judged_path.exists():
        return False
    n = 0
    for row in read_jsonl(judged_path):
        if not (row.get("judge") or "").strip():
            return False
        n += 1
    return n == expected_n


def process_file(file_path: Path, judge: IntrospectionJudge, out_prefix: str = "judged_") -> Path:

    judged_path = file_path.parent / f"{out_prefix}{file_path.name}"

    rows = list(read_jsonl(file_path))
    if _judged_file_complete(judged_path, len(rows)):
        print(f"[skip] {judged_path.name} already complete ({len(rows)} rows)")
        return judged_path

    victim_hf_id = _victim_hf_id_from_filename(file_path.name)
    victim_tokenizer = _get_victim_tokenizer(victim_hf_id) if victim_hf_id else None
    if victim_hf_id and victim_tokenizer is None:
        print(f"[warn] {file_path.name}: tokenizer unavailable for {victim_hf_id}, using char fallback")

    results = [None] * len(rows)

    def _judge_row(i, row):
        response_text = row.get("followup_response", "")
        followup_id = row.get("followup_id") or ""
        if not followup_id or not response_text:
            return i, {**row, "judge": "ERROR", "judge_model": judge.client.model_name}
        try:
            judgement = judge.evaluate(response_text, followup_id, victim_tokenizer=victim_tokenizer)
        except Exception as e:
            judgement = "ERROR"
        return i, {**row, "judge": judgement, "judge_model": judge.client.model_name}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_judge_row, i, row): i for i, row in enumerate(rows)}
        for fut in tqdm(as_completed(futures), total=len(rows), desc=f"Judging {file_path.name}"):
            i, result_row = fut.result()
            results[i] = result_row
            
    # Save the entire batch at once using our util
    # (Since it's a list of dicts, we loop over it)
    if judged_path.exists():
        judged_path.unlink()
    for row in results:
        write_jsonl(judged_path, row)
        
    return judged_path

def summarize_results(judged_files: list):
    all_rows = []
    for p in judged_files:
        all_rows.extend(read_jsonl(p))
        
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data to summarize.")
        return
        
    if "model_name" not in df.columns and "model" in df.columns:
        df["model_name"] = df["model"]
    if "model_name" not in df.columns:
        # Newer gen-file schema doesn't embed model id; skip the per-model summary.
        print("[summary] gen schema has no model_name; skipping per-model breakdown")
        return
        
    df["judge_answer"] = df["judge"].astype(str).str.upper().str.strip()
    valid = df[df["judge_answer"].isin(["YES", "NO"])].copy()
    
    summary = (
        valid.groupby(["model_name", "condition"])["judge_answer"]
        .value_counts(normalize=True)
        .rename("rate")
        .reset_index()
        .pivot_table(index=["model_name", "condition"], columns="judge_answer", values="rate", fill_value=0.0)
        .reset_index()
        .rename(columns={"YES": "yes_rate", "NO": "no_rate"})
    )
    
    print("\n=== FINAL AGGREGATED RESULTS ===")
    print(summary.to_string(index=False))

def run_judge_pipeline(judge_model: str, target_dataset: str, target_model: str, out_prefix: str = "judged_"):
    load_dotenv(override=True)
    client = OpenRouterClient(judge_model, os.environ.get("OPENROUTER_API_KEY"))
    judge = IntrospectionJudge(client)

    target_files = list(OUT_DIR.glob("gen_*.jsonl"))
    target_files = [f for f in target_files if not f.name.startswith("judged")]

    # Apply optional command line filters
    if target_dataset:
        target_files = [f for f in target_files if target_dataset in f.name]
    if target_model:
        safe_mname = target_model.replace("/", "__")
        target_files = [f for f in target_files if safe_mname in f.name]

    if not target_files:
        print("No generation files found to judge based on the given filters.")
        return

    print(f"[*] Found {len(target_files)} files to judge using {judge_model}. Output prefix: {out_prefix!r}")

    judged_files = []
    for file_path in target_files:
        judged_path = process_file(file_path, judge, out_prefix=out_prefix)
        judged_files.append(judged_path)

    summarize_results(judged_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Introspection Judge")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-4.1", help="Model to use for judging")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by specific dataset (e.g., harmbench)")
    parser.add_argument("--model", type=str, default=None, help="Filter by specific target model generations")
    parser.add_argument("--out-prefix", type=str, default="judged_", help="Prefix for output JSONL files (e.g., 'judged_'). Default preserves old behavior.")

    args = parser.parse_args()
    run_judge_pipeline(args.judge_model, args.dataset, args.model, args.out_prefix)
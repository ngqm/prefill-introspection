"""Judge gen_*.jsonl follow-up replies with an LLM and write judged_*.jsonl.

These labels stratify the classifier training sample: build_train_sample draws
equal numbers of YES and NO replies by this judge, and label_with_gpt41 then
relabels that sample with GPT-4.1. The default judge is DeepSeek V3.2.

For each generation file under OUT_DIR, run the IntrospectionJudge on every
row's followup_response and write <out-prefix><original-name> with `judge`
(YES, NO, or an error label) and `judge_model` fields. A rerun keeps the YES
and NO rows of an existing judged file and retries the rest. Replies are
truncated with the victim model's tokenizer, parsed from the filename. Needs
OPENROUTER_API_KEY.

Usage:
  python -m scripts.classify.run_judge [--judge-model ID] [--dataset D] [--model M]
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from core.config import OUT_DIR
from core.utils import read_jsonl, write_jsonl
from llm.openrouter import OpenRouterClient
from experiments.judge_evaluator import IntrospectionJudge
from scripts.classify.common import victim_hf_id_from_filename, get_victim_tokenizer

DEFAULT_JUDGE = "deepseek/deepseek-v3.2"
MAX_WORKERS = 96
LABELS = ("YES", "NO")


def process_file(file_path: Path, judge: IntrospectionJudge, out_prefix: str = "judged_") -> Path:
    """Judge the rows of one generation file that lack a YES/NO label and write the judged file.

    Return the judged file's path.
    """
    judged_path = file_path.parent / f"{out_prefix}{file_path.name}"
    rows = list(read_jsonl(file_path))
    results = list(read_jsonl(judged_path))
    if len(results) != len(rows):
        results = [None] * len(rows)
    todo = [i for i, r in enumerate(results) if r is None or r.get("judge") not in LABELS]
    if not todo:
        print(f"[skip] {judged_path.name} already complete ({len(rows)} rows)")
        return judged_path

    victim_hf_id = victim_hf_id_from_filename(file_path.name)
    victim_tokenizer = get_victim_tokenizer(victim_hf_id) if victim_hf_id else None
    if victim_hf_id and victim_tokenizer is None:
        print(f"[warn] {file_path.name}: tokenizer unavailable for {victim_hf_id}, using char fallback")

    def _judge_row(i):
        row = rows[i]
        response_text = row.get("followup_response", "")
        followup_id = row.get("followup_id") or ""
        judgement = "ERROR"
        if followup_id and response_text:
            try:
                judgement = judge.evaluate(response_text, followup_id, victim_tokenizer=victim_tokenizer)
            except Exception:
                pass
        return i, {**row, "judge": judgement, "judge_model": judge.client.model_name}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_judge_row, i) for i in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Judging {file_path.name}"):
            i, result_row = fut.result()
            results[i] = result_row

    n_failed = sum(1 for r in results if r["judge"] not in LABELS)
    if n_failed:
        print(f"[warn] {judged_path.name}: {n_failed} rows without a YES/NO label; rerun to retry")
    if judged_path.exists():
        judged_path.unlink()
    for row in results:
        write_jsonl(judged_path, row)
    return judged_path


def summarize_results(judged_files: list):
    """Print the YES/NO rate per (model_name, condition) over the judged files."""
    df = pd.DataFrame([row for p in judged_files for row in read_jsonl(p)])
    if df.empty or "model_name" not in df.columns:
        print("No data to summarize.")
        return

    df["judge_answer"] = df["judge"].astype(str).str.upper().str.strip()
    valid = df[df["judge_answer"].isin(LABELS)].copy()
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
    """Judge all gen_*.jsonl files under OUT_DIR that match the filters, then summarize."""
    load_dotenv(override=True)
    client = OpenRouterClient(judge_model, os.environ.get("OPENROUTER_API_KEY"))
    judge = IntrospectionJudge(client)

    target_files = list(OUT_DIR.glob("gen_*.jsonl"))
    if target_dataset:
        target_files = [f for f in target_files if target_dataset in f.name]
    if target_model:
        safe_mname = target_model.replace("/", "__")
        target_files = [f for f in target_files if safe_mname in f.name]
    if not target_files:
        print("No generation files found to judge based on the given filters.")
        return

    print(f"[*] Found {len(target_files)} files to judge using {judge_model}. Output prefix: {out_prefix!r}")
    judged_files = [process_file(f, judge, out_prefix=out_prefix) for f in target_files]
    summarize_results(judged_files)


def main():
    """Parse arguments and run the judge."""
    parser = argparse.ArgumentParser(description="Judge follow-up replies with an LLM via OpenRouter.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE,
                        help=f"OpenRouter model id (default {DEFAULT_JUDGE})")
    parser.add_argument("--dataset", default=None, help="Filter by dataset substring (e.g., harmbench)")
    parser.add_argument("--model", default=None, help="Filter by victim model HF id")
    parser.add_argument("--out-prefix", default="judged_", help="Prefix for output JSONL files")
    args = parser.parse_args()
    run_judge_pipeline(args.judge_model, args.dataset, args.model, args.out_prefix)


if __name__ == "__main__":
    main()

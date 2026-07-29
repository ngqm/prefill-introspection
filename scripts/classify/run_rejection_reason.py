"""Add rejection_category and rejection_summary to judged_gen_*.jsonl rows where judge=NO.

Run the RejectionReasonJudge on each NO reply and write the resulting
rejection_category (SAFETY|TRUNCATION|OTHER) and rejection_summary back into
the file in place (atomic replace via a .tmp file). Idempotent: rows that
already have a rejection_category are skipped. For HarmBench files, only rows
whose type is 'standard' are in scope; SocialHarmBench is fully in scope.

Command-line arguments:
  --judge-model  OpenRouter model id (default 'openai/gpt-4.1-mini')
  --pattern      glob inside rq1_runs/ (default 'judged_gen_*_f1.jsonl')
  --filter       substring filter on filenames
  --smoke        if >0, sample this many NO rows per file and only PRINT (no write)
  --limit-files  process at most this many files (0 = all)

Reads OPENROUTER_API_KEY from the environment (via python-dotenv).
"""

import os
import json
import time
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dotenv import load_dotenv
from tqdm.auto import tqdm

from core.config import OUT_DIR
from core.utils import write_jsonl
from llm.openrouter import OpenRouterClient
from experiments.judge_evaluator import RejectionReasonJudge

DEFAULT_JUDGE = "openai/gpt-4.1-mini"
MAX_WORKERS = 32


def iter_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file, skipping blank lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def needs_judging(row):
    """Return True if the row is a NO verdict with no rejection_category yet."""
    return (row.get("judge") or "").upper().strip() == "NO" and "rejection_category" not in row


def _is_in_scope(path_name: str, row) -> bool:
    """Match the standard-only HarmBench policy used elsewhere in the project.
    SocialHarmBench has no functional-category split, so all of it is in scope."""
    if "harmbench" in path_name.lower() and "socialharmbench" not in path_name.lower():
        return row.get("type") == "standard"
    return True


def process_file(path: Path, judge: RejectionReasonJudge, dry_run: bool = False, sample: int = 0):
    """Judge the in-scope NO rows of one file and rewrite it in place.

    Optionally restrict to a seeded random sample of the targets. When dry_run
    is set, judge without writing back and return a few sample rows for
    inspection. Return a summary dict with counts and the category tally.
    """
    rows = list(iter_jsonl(path))
    targets = [(i, r) for i, r in enumerate(rows) if needs_judging(r) and _is_in_scope(path.name, r)]

    if sample and targets:
        random.seed(42)
        targets = random.sample(targets, min(sample, len(targets)))

    if not targets:
        return {"file": str(path), "n_judged": 0, "n_total_no": 0}

    def _judge(i, r):
        verdict = judge.evaluate(r.get("followup_response", ""), r.get("followup_text", ""))
        return i, {**r, "rejection_category": verdict["category"], "rejection_summary": verdict["summary"]}

    out = list(rows)
    cats = Counter()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_judge, i, r) for i, r in targets]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"reason {path.name}"):
            i, new_row = fut.result()
            cats[new_row["rejection_category"]] += 1
            out[i] = new_row

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        for r in out:
            write_jsonl(tmp, r)
        os.replace(tmp, path)

    return {
        "file": str(path),
        "n_judged": len(targets),
        "n_total_no": sum(1 for r in rows if (r.get("judge") or "").upper().strip() == "NO"),
        "categories": dict(cats),
        "samples": [out[i] for i, _ in targets[:5]] if dry_run else [],
    }


def main():
    """Parse arguments and run the rejection-reason judge over the matching files."""
    parser = argparse.ArgumentParser(description="Run rejection-reason judge on judged_gen_*.jsonl rows where judge=NO")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--pattern", default="judged_gen_*_f1.jsonl", help="glob inside rq1_runs/ (default: F1 files only)")
    parser.add_argument("--filter", default=None, help="substring filter on filenames (e.g. 'gemma-2-9b-it')")
    parser.add_argument("--smoke", type=int, default=0, help="if >0, sample this many NO rows per file and only PRINT (no write)")
    parser.add_argument("--limit-files", type=int, default=0, help="if >0, only process this many files")
    args = parser.parse_args()

    load_dotenv(override=True)
    client = OpenRouterClient(args.judge_model, os.environ.get("OPENROUTER_API_KEY"))
    judge = RejectionReasonJudge(client)

    files = sorted(OUT_DIR.glob(args.pattern))
    if args.filter:
        files = [f for f in files if args.filter in f.name]
    if args.limit_files:
        files = files[:args.limit_files]

    if not files:
        print("No files matched.")
        return

    mode = f"SMOKE TEST (sample={args.smoke})" if args.smoke else "FULL"
    print(f"[*] {mode}  judge={args.judge_model}  files={len(files)}")
    t0 = time.time()
    overall = Counter()
    for fp in files:
        info = process_file(fp, judge, dry_run=bool(args.smoke), sample=args.smoke)
        if info["n_judged"] == 0:
            continue
        overall.update(info.get("categories", {}))
        print(f"  {fp.name}: judged {info['n_judged']}/{info['n_total_no']} NO rows  cats={info.get('categories')}")
        for s in info.get("samples", [])[:3]:
            print(f"    [{s['rejection_category']:10s}] {s['rejection_summary']}")
            fr = (s.get("followup_response") or "")[:140].replace("\n", " ")
            print(f"      ↳ followup_response: {fr}")
    dt = time.time() - t0
    print(f"\n[*] done in {dt:.1f}s  totals={dict(overall)}")


if __name__ == "__main__":
    main()

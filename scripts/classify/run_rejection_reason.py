"""Augments existing judged_gen_*.jsonl rows that have judge=NO with a
rejection_category (SAFETY|TRUNCATION|OTHER) and rejection_summary field.

Idempotent: rows that already have rejection_category are skipped.

Smoke-test mode samples a few rows and prints results without writing back,
so cost can be sanity-checked before the full run.
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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def needs_judging(row):
    return (row.get("judge") or "").upper().strip() == "NO" and "rejection_category" not in row


def _is_in_scope(path_name: str, row) -> bool:
    """Match the standard-only HarmBench policy used elsewhere in the project.
    SocialHarmBench has no functional-category split, so all of it is in scope."""
    if "harmbench" in path_name.lower() and "socialharmbench" not in path_name.lower():
        return row.get("type") == "standard"
    return True


def process_file(path: Path, judge: RejectionReasonJudge, dry_run: bool = False, sample: int = 0):
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

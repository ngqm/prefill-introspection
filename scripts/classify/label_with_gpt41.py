"""Label the 3000-row F1 or F2 sample with GPT-4.1 using the probe's judge
prompt. Adds a judge_gpt41 field; skips already-labeled rows for resume.

Usage: python -m scripts.classify.label_with_gpt41 --probe {f1,f2}
"""
import json, os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm.auto import tqdm

from llm.openrouter import OpenRouterClient
from experiments.judge_evaluator import IntrospectionJudge

FOLLOWUP_ID = {"f1": "F1_minimal", "f2": "F2_tamper_check"}

parser = argparse.ArgumentParser()
parser.add_argument("--probe", choices=["f1", "f2"], required=True)
args = parser.parse_args()

load_dotenv(override=True)
ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / f"{args.probe}_classifier" / "sample_3000.jsonl"
OUT_PATH = ROOT / f"{args.probe}_classifier" / "sample_3000_gpt41.jsonl"

items = [json.loads(l) for l in IN_PATH.open()]

# Resume: load any already-labeled rows.
existing = {}
if OUT_PATH.exists():
    for line in OUT_PATH.open():
        r = json.loads(line)
        if r.get("judge_gpt41") in ("YES", "NO", "ERROR"):
            existing[r["id"]] = r
print(f"items: {len(items)}; already labeled: {len(existing)}; to do: {len(items)-len(existing)}")

client = OpenRouterClient("openai/gpt-4.1", os.environ.get("OPENROUTER_API_KEY"), max_retries=3)
judge = IntrospectionJudge(client)

# Use the pre-truncated text so we feed exactly what the judge prompt would see
# normally. IntrospectionJudge.evaluate also truncates if a tokenizer is passed;
# since we already truncated, pass tokenizer=None to skip the second pass.
def _go(item):
    try:
        label = judge.evaluate(item["text_truncated"], FOLLOWUP_ID[args.probe], victim_tokenizer=None)
    except Exception as e:
        label = f"ERROR:{type(e).__name__}"
    return {**item, "judge_gpt41": label}

todo = [it for it in items if it["id"] not in existing]
done_buf = list(existing.values())

with ThreadPoolExecutor(max_workers=16) as ex, OUT_PATH.open("a") as out:
    if not existing:
        out.truncate(0)
    futs = [ex.submit(_go, it) for it in todo]
    for fut in tqdm(as_completed(futs), total=len(futs), desc="gpt-4.1 labeling"):
        r = fut.result()
        out.write(json.dumps(r) + "\n")
        out.flush()
        done_buf.append(r)

from collections import Counter
c = Counter(r["judge_gpt41"] for r in done_buf)
print(f"\nGPT-4.1 label distribution: {dict(c)}")

# Compare to the judge labels
judge_vs_gpt = Counter()
for r in done_buf:
    judge_vs_gpt[(r["judge"], r["judge_gpt41"])] += 1
print(f"\nconfusion (rows: judge, cols: gpt-4.1):")
print(f"         GPT=YES   GPT=NO   GPT=ERROR")
print(f"  judge=YES {judge_vs_gpt[('YES','YES')]:7d}  {judge_vs_gpt[('YES','NO')]:7d}  {sum(v for k,v in judge_vs_gpt.items() if k[0]=='YES' and k[1].startswith('ERROR')):8d}")
print(f"  judge=NO  {judge_vs_gpt[('NO','YES')]:7d}  {judge_vs_gpt[('NO','NO')]:7d}  {sum(v for k,v in judge_vs_gpt.items() if k[0]=='NO' and k[1].startswith('ERROR')):8d}")
print(f"\nwrote {OUT_PATH}")

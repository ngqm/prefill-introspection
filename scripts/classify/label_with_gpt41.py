"""Label the 3,000-reply F1 or F2 classifier sample with GPT-4.1.

Reads <probe>_classifier/sample_3000.jsonl from build_train_sample and writes
sample_3000_gpt41.jsonl with a judge_gpt41 field. A rerun keeps the YES/NO
rows and retries the rest. Needs OPENROUTER_API_KEY.

Usage:
  python -m scripts.classify.label_with_gpt41 --probe {f1,f2}
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm.auto import tqdm

from core.config import ROOT_DIR
from core.utils import read_jsonl
from llm.openrouter import OpenRouterClient
from experiments.judge_evaluator import IntrospectionJudge

FOLLOWUP_ID = {"f1": "F1_minimal", "f2": "F2_tamper_check"}
LABELS = ("YES", "NO")


def main():
    """Label the unlabeled sample rows with GPT-4.1 and print agreement with the stratification judge."""
    parser = argparse.ArgumentParser(description="Label the classifier training sample with GPT-4.1.")
    parser.add_argument("--probe", choices=["f1", "f2"], required=True)
    args = parser.parse_args()

    load_dotenv(override=True)
    in_path = ROOT_DIR / f"{args.probe}_classifier" / "sample_3000.jsonl"
    out_path = ROOT_DIR / f"{args.probe}_classifier" / "sample_3000_gpt41.jsonl"

    items = list(read_jsonl(in_path))
    done = {r["id"]: r for r in read_jsonl(out_path) if r.get("judge_gpt41") in LABELS}
    todo = [it for it in items if it["id"] not in done]
    print(f"items: {len(items)}; already labeled: {len(done)}; to do: {len(todo)}")

    client = OpenRouterClient("openai/gpt-4.1", os.environ.get("OPENROUTER_API_KEY"), max_retries=3)
    judge = IntrospectionJudge(client)

    # text_truncated is already cut to 64 victim tokens. With no tokenizer,
    # evaluate() also applies its 256-character fallback cap.
    def _go(item):
        try:
            label = judge.evaluate(item["text_truncated"], FOLLOWUP_ID[args.probe], victim_tokenizer=None)
        except Exception as e:
            label = f"ERROR:{type(e).__name__}"
        return {**item, "judge_gpt41": label}

    results = list(done.values())
    with out_path.open("w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")
        out.flush()
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_go, it) for it in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="gpt-4.1 labeling"):
                r = fut.result()
                if r["judge_gpt41"] in LABELS:
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                results.append(r)

    labeled = [r for r in results if r["judge_gpt41"] in LABELS]
    n_failed = len(results) - len(labeled)
    print(f"\nGPT-4.1 labels: {dict(Counter(r['judge_gpt41'] for r in labeled))}")
    if n_failed:
        print(f"[warn] {n_failed} rows failed and were not written; rerun to retry")

    judge_name = labeled[0].get("judge_model", "run_judge") if labeled else "run_judge"
    pairs = Counter((r["judge"], r["judge_gpt41"]) for r in labeled)
    print(f"\nconfusion (rows: {judge_name}, cols: gpt-4.1):")
    print("            GPT=YES   GPT=NO")
    for j in LABELS:
        print(f"  judge={j:<3s} {pairs[(j, 'YES')]:7d}  {pairs[(j, 'NO')]:7d}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

"""Sample follow-up replies for the human-validation kappa.

Draws a stratified random sample (default 200 per probe) from the classified
corpus, balanced across model family and across control versus prefilled
conditions. Writes rq1_runs/annotation_sample_<probe>.jsonl with the fields
the annotator needs and none of the judge or classifier labels.

Usage:
  python -m scripts.classify.sample_for_annotation [--n 200] [--seed 42] [--probes F1,F2]
"""
import argparse
import json
import random

from core.config import OUT_DIR
from core.utils import read_jsonl

# Annotated models and their family, the stratification unit.
FAMILY = {
    "meta-llama__Llama-3.1-8B-Instruct": "Llama",
    "Qwen__Qwen3-8B": "Qwen",
    "Qwen__Qwen3-14B": "Qwen",
    "google__gemma-3-12b-it": "Gemma",
}
DATASETS = ["harmbench", "socialharmbench", "jailbreakbench", "strongreject"]


def reply_id(model, r):
    """Return a stable reply id joining model, prompt, condition, prefill, and followup ids."""
    return "::".join([model, r.get("prompt_id", ""), r.get("condition") or "",
                      r.get("prefill_id") or "", r.get("followup_id") or ""])


def collect_pool(probe):
    """Yield (reply_id, slim_row) for every classified reply of the annotated models."""
    suffix = probe.lower()
    for model in FAMILY:
        for ds in DATASETS:
            for adv in (False, True):
                path = OUT_DIR / f"classified_gen{'_adv' if adv else ''}_{model}_{ds}_{suffix}.jsonl"
                for r in read_jsonl(path):
                    if ds == "harmbench" and r.get("type") and r.get("type") != "standard":
                        continue
                    cond = r.get("condition")
                    if cond not in ("control", "adv_prefill", "prefill"):
                        continue
                    rid = reply_id(model, r)
                    yield rid, {
                        "reply_id": rid,
                        "model_name": model,
                        "dataset": ds,
                        "prompt_id": r.get("prompt_id"),
                        "prompt_text": r.get("prompt_text"),
                        "condition": cond,
                        "prefill_id": r.get("prefill_id"),
                        "prefill_text": r.get("prefill_text"),
                        "initial_response": r.get("initial_response"),
                        "followup_id": r.get("followup_id"),
                        "followup_text": r.get("followup_text"),
                        "followup_response": r.get("followup_response"),
                    }


def stratified_sample(probe, n, seed=42):
    """Return n replies stratified by (family, control vs prefilled).

    Draw an even share per bucket, then top up with a seeded random sample of
    the remaining pool if the even shares undershoot n.
    """
    pool = dict(collect_pool(probe))
    print(f"[{probe}] pool size: {len(pool)}")

    buckets = {}
    for rid, row in pool.items():
        cb = "control" if row["condition"] == "control" else "prefilled"
        buckets.setdefault((FAMILY[row["model_name"]], cb), []).append(rid)

    rng = random.Random(seed)
    keys = sorted(buckets)
    per_bucket = max(1, n // len(keys))
    picked = []
    for k in keys:
        ids = buckets[k]
        rng.shuffle(ids)
        picked.extend(ids[:per_bucket])
    if len(picked) < n:
        taken = set(picked)
        rest = [rid for rid in pool if rid not in taken]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    picked = picked[:n]
    rng.shuffle(picked)
    return [pool[rid] for rid in picked]


def main():
    """Write an annotation sample file per requested probe."""
    p = argparse.ArgumentParser(description="Sample replies for human annotation.")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probes", default="F1,F2")
    args = p.parse_args()

    for probe in args.probes.split(","):
        probe = probe.strip()
        sample = stratified_sample(probe, args.n, seed=args.seed)
        out = OUT_DIR / f"annotation_sample_{probe.lower()}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in sample))
        print(f"[ok] {probe}: wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    main()

"""Sample replies for the human-validation κ appendix.

Draws a stratified random sample (default 200) from the full judged corpus per
probe (F1, F2), balanced across condition (control vs prefilled) and across
the three primary model families.

Writes rq1_runs/annotation_sample_{f1,f2}.jsonl with the minimum fields the
annotator needs (and explicitly *not* the classifier_label / judge labels).
"""
import argparse
import json
import random
from pathlib import Path

PRIMARY = [
    "meta-llama__Llama-3.1-8B-Instruct",
    "Qwen__Qwen3-8B",
    "Qwen__Qwen3-14B",
    "google__gemma-3-12b-it",
]
DATASETS = ["harmbench", "socialharmbench", "jailbreakbench", "strongreject"]
RQ = Path("rq1_runs")


def reply_id(r):
    return "::".join([
        r.get("model_name", ""),
        r.get("prompt_id", ""),
        r.get("condition", "") or "",
        r.get("prefill_id", "") or "",
        r.get("followup_id", "") or "",
    ])


def collect_pool(probe):
    """Yield (reply_id, slim_row) for every judged row across primary models
    and datasets, on both control and adv-prefill conditions.

    The model_name field on disk is None for many rows; we derive it from the
    filename instead, which is reliable across the run."""
    suffix = "f1" if probe == "F1" else "f2"
    for safe in PRIMARY:
        for ds in DATASETS:
            for adv in (False, True):
                path = RQ / f"classified_gen{'_adv' if adv else ''}_{safe}_{ds}_{suffix}.jsonl"
                if not path.exists():
                    continue
                for line in path.open():
                    r = json.loads(line)
                    if ds == "harmbench" and r.get("type") and r.get("type") != "standard":
                        continue
                    cond = r.get("condition")
                    if cond not in ("control", "adv_prefill", "prefill"):
                        continue
                    # Patch model_name from filename so reply_id is stable.
                    row = {**r, "model_name": safe}
                    yield reply_id(row), {
                        "reply_id": reply_id(row),
                        "model_name": safe,
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
    pool = {rid: row for rid, row in collect_pool(probe)}
    print(f"[{probe}] pool size: {len(pool)}")

    # Stratify by (family, condition_bucket) where family ∈ {Llama, Qwen, Gemma}
    # and condition_bucket ∈ {control, prefilled}.
    family_of = {
        "meta-llama__Llama-3.1-8B-Instruct": "Llama",
        "Qwen__Qwen3-8B": "Qwen",
        "Qwen__Qwen3-14B": "Qwen",
        "google__gemma-3-12b-it": "Gemma",
    }
    buckets = {}
    for rid, row in pool.items():
        fam = family_of.get(row["model_name"])
        if not fam:
            continue
        cb = "control" if row["condition"] == "control" else "prefilled"
        buckets.setdefault((fam, cb), []).append(rid)

    rng = random.Random(seed)
    keys = sorted(buckets)
    per_bucket = max(1, n // len(keys))
    picked = []
    for k in keys:
        ids = buckets[k]
        rng.shuffle(ids)
        picked.extend(ids[:per_bucket])
    # Top up by random sampling without replacement if we undershot.
    if len(picked) < n:
        rest = list(set(pool) - set(picked))
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    picked = picked[:n]
    rng.shuffle(picked)
    return [pool[rid] for rid in picked]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probes", default="F1,F2")
    args = p.parse_args()

    for probe in args.probes.split(","):
        probe = probe.strip()
        sample = stratified_sample(probe, args.n, seed=args.seed)
        out = RQ / f"annotation_sample_{probe.lower()}.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in sample) + "\n")
        print(f"[ok] {probe}: wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    main()

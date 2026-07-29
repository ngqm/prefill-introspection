"""Merges guard_label / guard_category from gen_*.jsonl into the matching
judged_gen_*.jsonl rows. Idempotent: skips judged files that already carry
guard_label on at least one row.

Why this is needed: the introspection judge step produced judged_gen_*.jsonl
files BEFORE Llama-Guard added guard_label to the gen files. Plotting code
that relies on guard_label silently dropped every row from judged files,
which is why the *_guard_*_quadrants_with_counts plots came out empty.
"""

import json
import os
from pathlib import Path

RUNS = Path("rq1_runs")


def _row_key(r):
    """Stable per-row key matching one gen row to one judged row.

    followup_id is intentionally omitted: each gen/judged file pair has a
    single followup_id, so it's redundant for matching, and historically the
    judged files were written with a slightly different label
    (F_tamper_check vs F2_tamper_check) which would otherwise blow up the
    match.
    """
    return (r.get("prompt_id"), r.get("condition"), r.get("prefill_id"))


def merge_one(gen_path: Path) -> dict:
    judged_path = gen_path.parent / f"judged_{gen_path.name}"
    if not judged_path.exists():
        return {"file": str(gen_path), "skipped": "no judged file"}

    # Build (key) -> (guard_label, guard_category) map from gen file.
    label_by_key = {}
    cat_by_key = {}
    has_guard = False
    with gen_path.open() as f:
        for line in f:
            r = json.loads(line)
            if "guard_label" in r:
                has_guard = True
                k = _row_key(r)
                label_by_key[k] = r.get("guard_label")
                cat_by_key[k] = r.get("guard_category", "")
    if not has_guard:
        return {"file": str(gen_path), "skipped": "gen file has no guard_label"}

    # Rewrite judged file with guard_label added.
    rows_in = []
    already_has = 0
    matched = 0
    unmatched = 0
    with judged_path.open() as f:
        for line in f:
            r = json.loads(line)
            rows_in.append(r)
            if "guard_label" in r:
                already_has += 1

    if already_has == len(rows_in):
        return {"file": str(judged_path), "skipped": "already merged", "n": already_has}

    rows_out = []
    for r in rows_in:
        if "guard_label" not in r:
            k = _row_key(r)
            if k in label_by_key:
                r = {**r, "guard_label": label_by_key[k], "guard_category": cat_by_key[k]}
                matched += 1
            else:
                unmatched += 1
        rows_out.append(r)

    tmp = judged_path.with_suffix(judged_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, judged_path)
    return {"file": str(judged_path), "matched": matched, "unmatched": unmatched, "total": len(rows_in)}


def main():
    gen_files = sorted(RUNS.glob("gen_*.jsonl"))
    gen_files = [f for f in gen_files if not f.name.startswith("judged_")]
    print(f"[*] {len(gen_files)} gen files to inspect")
    n_done = n_skipped = 0
    total_matched = 0
    for fp in gen_files:
        info = merge_one(fp)
        if "skipped" in info:
            n_skipped += 1
            if info["skipped"] != "already merged" and info["skipped"] != "no judged file":
                print(f"  skip {fp.name}: {info['skipped']}")
        else:
            n_done += 1
            total_matched += info["matched"]
            if info["unmatched"]:
                print(f"  WARN {fp.name}: matched={info['matched']} unmatched={info['unmatched']} (will leave those rows without guard_label)")
    print(f"[*] done: merged={n_done}, skipped={n_skipped}, total rows updated with guard_label={total_matched}")


if __name__ == "__main__":
    main()

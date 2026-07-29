"""Shared output routing for the generation entrypoints.

save_results() is used verbatim by run_adv, run_static, run_control_only, and
run_placebo_static_strings, which all emit the same
gen[_adv|_placebo]_<safe_mname>_<dataset>_f{1,2,3,4}.jsonl layout.
"""
from core.config import OUT_DIR, FOLLOWUPS
from core.utils import write_jsonl

SUFFIX_MAP = {
    "F1_minimal": "_f1",
    "F2_tamper_check": "_f2",
    "F3_minimal_truncated": "_f3",
    "F4_tamper_check_truncated": "_f4",
}


def save_results(results: list, base_filename: str, dataset_name: str):
    """Route followups to the _f1, _f2, _f3, _f4 files.

    Truncate any existing target files, write error records to all four to keep
    line parity, and write each present followup to its matching suffix file.
    """
    out_paths = {}
    for f_id, suffix in SUFFIX_MAP.items():
        out_paths[f_id] = OUT_DIR / f"{base_filename}_{dataset_name}{suffix}.jsonl"
        if out_paths[f_id].exists():
            out_paths[f_id].unlink()

    for rec in results:
        if "error" in rec:
            # Write errors to all files to maintain line parity
            for path in out_paths.values():
                write_jsonl(path, rec)
            continue

        base_record = {k: v for k, v in rec.items() if k != "followups"}

        for f_id in SUFFIX_MAP:
            if f_id in rec.get("followups", {}):
                write_jsonl(out_paths[f_id], {
                    **base_record,
                    "followup_id": f_id,
                    "followup_text": FOLLOWUPS[f_id],
                    "followup_response": rec["followups"][f_id],
                })

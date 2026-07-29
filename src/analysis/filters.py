"""Shared filters for the canonical RQ1/RQ3 analysis subset.

Two filters operate together in the main-text analyses:
  1. CANONICAL_1085: restrict HarmBench to the official 159-standard split
     (per the methodology section), so the four-dataset total is 1,085.
  2. BEHAVIOR_CHANGE: keep only (model, prompt) rows where the prefill
     changes the model's safety behavior, i.e. the Llama-Guard-3-1B
     label on the control response differs from the label on the
     adv-prefilled response. Both directions are included
     (safe -> unsafe, the typical attack-success case, and the rarer
     unsafe -> safe).
"""
import csv
import os
from pathlib import Path


def load_canonical_harmbench_ids(csv_path: str = None):
    csv_path = csv_path or os.getenv("HARM_BENCH_CSV_PATH", "data/harmbench_behaviors_text_test.csv")
    with open(csv_path) as f:
        return {r["BehaviorID"] for r in csv.DictReader(f) if r["FunctionalCategory"] == "standard"}


_CANONICAL = None


def is_canonical(dataset: str, prompt_id: str) -> bool:
    """True iff (dataset, prompt_id) is in the canonical 1,085-prompt set.
    For non-HarmBench datasets this is always True (the JB / SR / SHB
    files on disk are already the filtered versions used in the paper).
    """
    global _CANONICAL
    if _CANONICAL is None:
        _CANONICAL = load_canonical_harmbench_ids()
    if str(dataset).lower() == "harmbench":
        return prompt_id in _CANONICAL
    return True


def is_behavior_change(ctrl_row, adv_row) -> bool:
    """True iff the guard label differs between control and adv-prefill.
    Returns False if either row lacks a guard_label (defensive default)."""
    cg = ctrl_row.get("guard_label")
    ag = adv_row.get("guard_label")
    if cg is None or ag is None:
        return False
    return cg != ag

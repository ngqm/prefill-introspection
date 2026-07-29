"""Filters for the canonical prompt set and the behavior-change subset.

The canonical set keeps the 159 standard HarmBench behaviors and every prompt
of the other three benchmarks, 1,085 prompts in all. The behavior-change
subset keeps the prompts whose Llama Guard 3 1B label differs between the
control and the prefilled response, in either direction.
"""
import csv

from core.config import HARM_BENCH_CSV


def load_canonical_harmbench_ids(csv_path=None):
    """Return the set of standard HarmBench BehaviorIDs.

    Reads csv_path when given, else core.config.HARM_BENCH_CSV.
    """
    with open(csv_path or HARM_BENCH_CSV) as f:
        return {r["BehaviorID"] for r in csv.DictReader(f) if r["FunctionalCategory"] == "standard"}


_CANONICAL = None


def is_canonical(dataset: str, prompt_id: str) -> bool:
    """Return True if (dataset, prompt_id) is in the canonical 1,085-prompt set.

    Only HarmBench is filtered; the other benchmark files already hold the
    paper's prompt sets.
    """
    global _CANONICAL
    if str(dataset).lower() != "harmbench":
        return True
    if _CANONICAL is None:
        _CANONICAL = load_canonical_harmbench_ids()
    return prompt_id in _CANONICAL


def is_behavior_change(ctrl_row, adv_row) -> bool:
    """Return True if the guard label differs between control and adversarial prefill.

    A row without a guard_label never counts as a change.
    """
    cg = ctrl_row.get("guard_label")
    ag = adv_row.get("guard_label")
    if cg is None or ag is None:
        return False
    return cg != ag

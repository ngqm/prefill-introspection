"""Run the adversarial-prefill introspection experiment.

AdvPrefillExperiment prefills each prompt with its optimized adversarial prefix,
generates the initial response, then asks each follow-up probe.
load_optimized_prefixes reads a model's prefix store.
"""
import json
from typing import List, Dict, Any
from core.config import PREFIX_DIR, PREFIX_MODEL_MAPPING
from llm.base import BaseLLMClient
from experiments.prefill_experiment import PrefillExperiment


def load_optimized_prefixes(model_name: str, dataset_name: str) -> Dict[str, List[str]]:
    """Return the {prompt_id: [prefix]} store for a base model and dataset.

    Read PREFIX_DIR/optimized_prefixes_<model>_<dataset>.json, falling back to
    the sibling store named in PREFIX_MODEL_MAPPING. Raise FileNotFoundError
    when neither exists.
    """
    prefix_file = PREFIX_DIR / f"optimized_prefixes_{model_name.replace('/', '__')}_{dataset_name}.json"
    if not prefix_file.exists():
        mapped = PREFIX_MODEL_MAPPING.get(model_name.lower())
        if mapped:
            fallback = PREFIX_DIR / f"optimized_prefixes_{mapped}_{dataset_name}.json"
            if fallback.exists():
                print(f"[*] {prefix_file.name} not found; using {fallback.name} per PREFIX_MODEL_MAPPING.")
                prefix_file = fallback
    if not prefix_file.exists():
        raise FileNotFoundError(
            f"{prefix_file} not found. Run scripts/generate/optimize_adv_prefixes.py first.")
    with open(prefix_file) as f:
        return json.load(f)


class AdvPrefillExperiment(PrefillExperiment):
    """Run adversarial-prefill trials and follow-up probes over a dataset.

    Takes the LLM client, the follow-up probe map, the per-prompt
    optimized_prefixes map, and an optional system_prompt prepended to every
    conversation.
    """

    def __init__(self, client: BaseLLMClient, followups: Dict[str, str], optimized_prefixes: Dict[str, List[str]],
                 system_prompt: str = None):
        super().__init__(client, followups, system_prompt)
        self.optimized_prefixes = optimized_prefixes

    def _jobs(self, row):
        """Yield the single adversarial-prefill job for row.

        A prompt missing from the prefix store runs with an empty prefill (as
        in the paper's runs) and a warning.
        """
        prompt_id = str(row["prompt_id"])
        prefill_list = self.optimized_prefixes.get(prompt_id, [""])
        prefill_text = prefill_list[0] if prefill_list else ""
        if not prefill_text:
            print(f"[!] prompt_id {prompt_id} has no optimized prefix; "
                  f"it runs with an empty prefill but is labelled adv_prefill", flush=True)
        yield "adv_prefill", prefill_text

    def _process_row(self, row: Dict[str, Any], condition_id: str, prefill_text: str):
        """Run one trial for row and wrap it with condition metadata.

        Return the row merged with condition='adv_prefill' and trial output on
        success, or condition='error' with the error string on failure.
        """
        try:
            trial_data = self.run_trial(row["prompt_text"], prefill_text)
            return {
                **row,
                "condition": "adv_prefill",
                "prefill_id": "adv_prefill",
                "prefill_text": prefill_text,
                **trial_data
            }
        except Exception as e:
            return {
                **row,
                "condition": "error",
                "prefill_text": prefill_text,
                "error": str(e)
            }

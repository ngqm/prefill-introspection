"""Run the static-prefill introspection experiment.

StaticPrefillExperiment sweeps each prompt through a no-prefill control plus
every fixed prefill in a prefill map, generating the initial response and each
follow-up probe. An empty prefill map runs the control alone.
"""
from typing import Dict, Any
from llm.base import BaseLLMClient
from experiments.prefill_experiment import PrefillExperiment


class StaticPrefillExperiment(PrefillExperiment):
    """Run static-prefill trials (control plus fixed prefills) over a dataset.

    Takes the LLM client, the follow-up probe map, the prefill_map of id to
    prefill literal, and an optional system_prompt prepended to every
    conversation.
    """

    def __init__(self, client: BaseLLMClient, followups: Dict[str, str], prefill_map: Dict[str, str],
                 system_prompt: str = None):
        super().__init__(client, followups, system_prompt)
        self.prefill_map = prefill_map

    def _jobs(self, row):
        """Yield the control job, then one job per static prefill."""
        yield "control", None
        yield from self.prefill_map.items()

    def _process_row(self, row: Dict[str, Any], condition_id: str, prefill_text: str):
        """Run one trial for row and wrap it with condition metadata.

        Control rows carry condition='control', prefill_id=None, and
        prefill_text=''; prefilled rows carry condition='prefill' with the
        variant id in prefill_id. Return condition='error' with the error
        string on failure.
        """
        try:
            trial_data = self.run_trial(row["prompt_text"], prefill_text)
            if condition_id == "control":
                meta = {"condition": "control", "prefill_id": None, "prefill_text": ""}
            else:
                meta = {"condition": "prefill", "prefill_id": condition_id, "prefill_text": prefill_text or ""}
            return {**row, **meta, **trial_data}
        except Exception as e:
            return {**row, "condition": "error", "error": str(e)}

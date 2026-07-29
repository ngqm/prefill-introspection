"""Shared base for the prefill introspection experiments.

PrefillExperiment runs one trial: generate the (optionally prefilled) initial
response, then query each follow-up probe. The adversarial-prefill and
static-prefill experiments subclass it and add their own prefill source and
sweep strategy.
"""
from typing import Dict, Any
from llm.base import BaseLLMClient


class PrefillExperiment:
    """Shared client/probe state and the single-trial runner.

    Takes the LLM client, the follow-up probe map, and an optional
    system_prompt prepended to every conversation. Subclasses add their own
    prefill source and sweep strategy.
    """

    def __init__(self, client: BaseLLMClient, followups: Dict[str, str],
                 system_prompt: str = None):
        self.client = client
        self.followups = followups
        self.system_prompt = system_prompt

    def _sys(self):
        """Return the system message list, or an empty list when no system prompt is set."""
        return [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

    def run_trial(self, prompt_text: str, prefill_text: str = None) -> Dict[str, Any]:
        """Run one trial: generate the prefilled initial response, then every follow-up.

        Prepend prefill_text to the assistant turn when given, re-inject it if
        the server drops it, and query each follow-up against the resulting
        response. Return a dict with initial_response and a followups map keyed
        by probe id.
        """
        messages = self._sys() + [{"role": "user", "content": prompt_text}]
        if prefill_text:
            messages.append({"role": "assistant", "content": prefill_text})

        init_response, _ = self.client.generate_chat(messages, max_tokens=2048)

        # Some servers swallow the prefill; enforce that it prefixes the stored text.
        if prefill_text and not init_response.startswith(prefill_text):
            init_response = prefill_text + init_response

        results = {"initial_response": init_response, "followups": {}}
        for f_id, f_text in self.followups.items():
            f_messages = self._sys() + [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": init_response},
                {"role": "user", "content": f_text},
            ]
            f_response, _ = self.client.generate_chat(f_messages, max_tokens=512)
            results["followups"][f_id] = f_response
        return results

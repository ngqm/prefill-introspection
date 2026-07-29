"""Run the adversarial-prefill introspection experiment.

AdvPrefillExperiment prefills each prompt with its optimized adversarial prefix,
generates the initial response, then asks each follow-up probe. run_sweep
drives the whole dataset with threaded concurrency and optional JSONL
checkpointing/resume.
"""
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from llm.base import BaseLLMClient
from experiments.prefill_experiment import PrefillExperiment

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

    def run_sweep(self, dataset: List[Dict[str, Any]], concurrency: int = 15, ckpt_path: str = None):
        """Run run_trial over every row in dataset with threaded concurrency.

        When ckpt_path is given, resume by skipping prompt_ids already recorded
        as non-error and append each new record to the JSONL file under a lock.
        Return the list of all records.
        """
        import json, os, threading
        all_records = []
        done = set()  # prompt_ids already completed (non-error)
        if ckpt_path and os.path.exists(ckpt_path):
            with open(ckpt_path) as ckpt_fh:
                for line in ckpt_fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("condition") == "error":
                        continue  # retry errors on resume
                    all_records.append(r)
                    done.add(str(r.get("prompt_id")))
            print(f"[resume] {len(all_records)} records already done in {ckpt_path}", flush=True)
        fh = open(ckpt_path, "a") if ckpt_path else None
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for row in dataset:
                prompt_id = str(row["prompt_id"])
                if prompt_id in done:
                    continue
                prefill_list = self.optimized_prefixes.get(prompt_id, [""])
                prefill_text = prefill_list[0] if prefill_list else ""

                futures.append(executor.submit(self._process_row, row, prefill_text))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Adv Sweep Progress"):
                rec = fut.result()
                all_records.append(rec)
                if fh:
                    with lock:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
        if fh:
            fh.close()
        return all_records

    def _process_row(self, row: Dict[str, Any], prefill_text: str):
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
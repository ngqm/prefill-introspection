"""Run the static-prefill introspection experiment.

StaticPrefillExperiment sweeps each prompt through a no-prefill control plus
every fixed prefill in a prefill map, generating the initial response and each
follow-up probe. run_sweep drives the dataset with threaded concurrency and
optional JSONL checkpointing/resume.
"""
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
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

    def run_sweep(self, dataset: List[Dict[str, Any]], concurrency: int = 10, ckpt_path: str = None):
        """Run the control and every prefill for each row with threaded concurrency.

        When ckpt_path is given, resume by skipping (prompt_id, condition) pairs
        already recorded as non-error and append each new record to the JSONL
        file under a lock. Return the list of all records.
        """
        import json, os, threading
        all_records = []
        done = set()  # (prompt_id, "control"|prefill_id) already completed (non-error)
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
                    cond = "control" if r.get("condition") == "control" else r.get("prefill_id")
                    done.add((str(r.get("prompt_id")), cond))
            print(f"[resume] {len(all_records)} records already done in {ckpt_path}", flush=True)
        fh = open(ckpt_path, "a") if ckpt_path else None
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for row in dataset:
                pid = str(row.get("prompt_id"))
                if (pid, "control") not in done:
                    futures.append(executor.submit(self._process_row, row, None, "control"))
                for p_id, p_text in self.prefill_map.items():
                    if (pid, p_id) not in done:
                        futures.append(executor.submit(self._process_row, row, p_text, p_id))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Sweep Progress"):
                rec = fut.result()
                all_records.append(rec)
                if fh:
                    with lock:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
        if fh:
            fh.close()
        return all_records

    def _process_row(self, row: Dict[str, Any], prefill_text: str, condition_id: str):
        """Run one trial for row and wrap it with condition metadata.

        condition_id is "control" for the no-prefill run, otherwise the prefill
        id. Return the row merged with the matching condition/prefill_id/
        prefill_text fields and trial output, or condition='error' with the
        error string on failure.
        """
        # Schema convention shared with adv_prefix.py + downstream plotting code:
        # control rows carry condition='control' / prefill_id=None / prefill_text='';
        # prefilled rows carry condition='prefill' with the variant id in
        # prefill_id (e.g. 'P1_affirmative') and the literal in prefill_text.
        # Keeping this alignment lets the plotting code key off prefill_id
        # uniformly across static + adv runs.
        try:
            trial_data = self.run_trial(row["prompt_text"], prefill_text)
            if condition_id == "control":
                meta = {"condition": "control", "prefill_id": None, "prefill_text": ""}
            else:
                meta = {"condition": "prefill", "prefill_id": condition_id, "prefill_text": prefill_text or ""}
            return {**row, **meta, **trial_data}
        except Exception as e:
            return {**row, "condition": "error", "error": str(e)}
"""Shared base for the prefill introspection experiments.

PrefillExperiment runs one trial (generate the optionally prefilled initial
response, then query each follow-up probe) and sweeps a dataset with threaded
concurrency and JSONL checkpoint/resume. Subclasses define which prefills each
prompt gets and how a finished trial becomes a record. save_results splits the
records into one file per follow-up probe.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm

from core.config import DEFAULT_CONCURRENCY, FOLLOWUPS
from core.utils import write_jsonl
from llm.base import BaseLLMClient

FOLLOWUP_SUFFIX = {"F1_minimal": "_f1", "F2_tamper_check": "_f2"}


class PrefillExperiment:
    """Shared client/probe state, the single-trial runner, and the sweep loop.

    Takes the LLM client, the follow-up probe map, and an optional
    system_prompt prepended to every conversation.
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

        Prepend prefill_text to the assistant turn when given, and store the
        prefill plus its continuation as the initial response. Return a dict
        with initial_response and a followups map keyed by probe id.
        """
        messages = self._sys() + [{"role": "user", "content": prompt_text}]
        if prefill_text:
            messages.append({"role": "assistant", "content": prefill_text})

        init_response, _ = self.client.generate_chat(messages, max_tokens=2048)

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

    def _jobs(self, row: Dict[str, Any]) -> Iterable[Tuple[str, Optional[str]]]:
        """Yield (condition_id, prefill_text) for every trial row needs."""
        raise NotImplementedError

    def _process_row(self, row: Dict[str, Any], condition_id: str,
                     prefill_text: Optional[str]) -> Dict[str, Any]:
        """Run one trial for row and return its record (condition='error' on failure)."""
        raise NotImplementedError

    def run_sweep(self, dataset: List[Dict[str, Any]], concurrency: int = DEFAULT_CONCURRENCY,
                  ckpt_path: str = None) -> List[Dict[str, Any]]:
        """Run every job of every row in dataset with threaded concurrency.

        When ckpt_path is given, append each finished record to that JSONL file
        and, on a rerun, reuse its non-error records for this sweep's jobs so
        only missing or failed jobs run again. Return the list of all records.
        """
        jobs = [(row, c_id, text) for row in dataset for c_id, text in self._jobs(row)]
        wanted = {(str(row["prompt_id"]), c_id) for row, c_id, _ in jobs}
        records, done = [], set()
        if ckpt_path and os.path.exists(ckpt_path):
            with open(ckpt_path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = _record_key(r)
                    if r.get("condition") == "error" or key not in wanted or key in done:
                        continue
                    records.append(r)
                    done.add(key)
            print(f"[resume] {len(records)} records already done in {ckpt_path}", flush=True)

        todo = [j for j in jobs if (str(j[0]["prompt_id"]), j[1]) not in done]
        with (open(ckpt_path, "a") if ckpt_path else nullcontext()) as fh, \
                ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self._process_row, *job) for job in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=type(self).__name__):
                rec = fut.result()
                records.append(rec)
                if fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
        return records


def _record_key(rec: Dict[str, Any]) -> Tuple[str, str]:
    """Return the (prompt_id, condition_id) job key a finished record belongs to."""
    return str(rec.get("prompt_id")), rec.get("prefill_id") or "control"


def save_results(results: List[Dict[str, Any]], out_dir: Path, base_filename: str,
                 dataset_name: str):
    """Write results to <out_dir>/<base_filename>_<dataset_name>_f{1,2}.jsonl.

    Replace existing files. Each record contributes one row per follow-up
    probe; error records are written to every file to keep line parity.
    """
    out_paths = {}
    for f_id, suffix in FOLLOWUP_SUFFIX.items():
        out_paths[f_id] = Path(out_dir) / f"{base_filename}_{dataset_name}{suffix}.jsonl"
        if out_paths[f_id].exists():
            out_paths[f_id].unlink()

    for rec in results:
        if "error" in rec:
            for path in out_paths.values():
                write_jsonl(path, rec)
            continue

        base_record = {k: v for k, v in rec.items() if k != "followups"}
        for f_id, path in out_paths.items():
            if f_id in rec.get("followups", {}):
                write_jsonl(path, {
                    **base_record,
                    "followup_id": f_id,
                    "followup_text": FOLLOWUPS[f_id],
                    "followup_response": rec["followups"][f_id],
                })

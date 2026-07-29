from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from llm.base import BaseLLMClient


class StaticPrefillExperiment:
    def __init__(self, client: BaseLLMClient, followups: Dict[str, str], prefill_map: Dict[str, str],
                 system_prompt: str = None):
        self.client = client
        self.followups = followups
        self.prefill_map = prefill_map
        self.system_prompt = system_prompt

    def _sys(self):
        return [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

    def run_trial(self, prompt_text: str, prefill_text: str = None) -> Dict[str, Any]:
        # 1. Initial Response
        messages = self._sys() + [{"role": "user", "content": prompt_text}]
        if prefill_text:
            messages.append({"role": "assistant", "content": prefill_text})

        init_response, _ = self.client.generate_chat(messages, max_tokens=2048)

        # Ensure prefill is included in the stored response text
        if prefill_text and not init_response.startswith(prefill_text):
            init_response = prefill_text + init_response

        # 2. Parallel Follow-ups
        results = {"initial_response": init_response, "followups": {}}

        for f_id, f_text in self.followups.items():
            f_messages = self._sys() + [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": init_response},
                {"role": "user", "content": f_text}
            ]
            f_response, _ = self.client.generate_chat(f_messages, max_tokens=512)
            results["followups"][f_id] = f_response
            
        return results

    def run_sweep(self, dataset: List[Dict[str, Any]], concurrency: int = 10, ckpt_path: str = None):
        import json, os, threading
        all_records = []
        done = set()  # (prompt_id, "control"|prefill_id) already completed (non-error)
        if ckpt_path and os.path.exists(ckpt_path):
            for line in open(ckpt_path):
                try:
                    r = json.loads(line)
                except Exception:
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
            # We wrap the logic to handle both Control (no prefill) and all Prefills
            futures = []
            for row in dataset:
                pid = str(row.get("prompt_id"))
                # Add Control
                if (pid, "control") not in done:
                    futures.append(executor.submit(self._process_row, row, None, "control"))
                # Add Prefills
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
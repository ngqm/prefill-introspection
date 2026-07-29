from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from llm.base import BaseLLMClient

class AdvPrefillExperiment:
    def __init__(self, client: BaseLLMClient, followups: Dict[str, str], optimized_prefixes: Dict[str, List[str]],
                 system_prompt: str = None):
        self.client = client
        self.followups = followups
        self.optimized_prefixes = optimized_prefixes
        self.system_prompt = system_prompt

    def _sys(self):
        return [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []

    def run_trial(self, prompt_text: str, prefill_text: str) -> Dict[str, Any]:
        # 1. Initial Response
        messages = self._sys() + [{"role": "user", "content": prompt_text}]
        if prefill_text:
            messages.append({"role": "assistant", "content": prefill_text})

        init_response, _ = self.client.generate_chat(messages, max_tokens=2048)

        # OpenRouter/vLLM servers sometimes swallow the prefill; this enforces it
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

    def run_sweep(self, dataset: List[Dict[str, Any]], concurrency: int = 15, ckpt_path: str = None):
        import json, os, threading
        all_records = []
        done = set()  # prompt_ids already completed (non-error)
        if ckpt_path and os.path.exists(ckpt_path):
            for line in open(ckpt_path):
                try:
                    r = json.loads(line)
                except Exception:
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
                # Fetch the top optimized prefix for this specific prompt
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
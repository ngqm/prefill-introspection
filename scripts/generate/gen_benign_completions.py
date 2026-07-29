"""Stage 1 of the placebo-prefill pipeline.

For each primary RQ3 model, generate a single completion to each prompt in
data/benign_prompts.txt and save the (prompt, response) pairs to
rq1_runs/benign_completions_<safe_mname>.jsonl.

Output one row per benign prompt:
  {"prompt_idx": int, "prompt": str, "response": str, "model": str}

The completions are stored in characters; downstream stages truncate to the
char-length of the corresponding adv_prefill prefix.
"""
import argparse
import json
from pathlib import Path

from core.config import OUT_DIR, VLLM_DEFAULT_PORT, resolve_model
import os as _os
from core.utils import set_seed
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager


def load_benign_prompts():
    p = Path("data/benign_prompts.txt")
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return lines


def run(model_name, tp, mem_fraction_static, port=None):
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated=False)
    out = OUT_DIR / f"benign_completions_{safe_mname}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[*] {out} already populated; skipping.")
        return

    prompts = load_benign_prompts()
    print(f"[*] Loaded {len(prompts)} benign prompts.")

    port = port if port is not None else int(_os.environ.get("VLLM_PORT", VLLM_DEFAULT_PORT))
    with VLLMServerManager(model_path, port=port,
                              tensor_parallel=tp,
                              mem_fraction_static=mem_fraction_static):
        client = VLLMClient(model_path, port=port)
        rows = []
        for i, prompt in enumerate(prompts):
            msgs = [{"role": "user", "content": prompt}]
            # Generate a long-enough response that we can truncate to any
            # adv_prefill length seen in the corpus (max ~250 chars).
            resp, _ = client.generate_chat(msgs, max_tokens=512)
            row = {
                "prompt_idx": i,
                "prompt": prompt,
                "response": resp,
                "model": model_name,
                "response_chars": len(resp),
            }
            rows.append(row)
            print(f"  [{i+1}/{len(prompts)}] chars={len(resp)} | {prompt[:60]}...")

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[ok] wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--mem-fraction-static", type=float, default=0.80)
    p.add_argument("--port", type=int, default=None,
                   help="vLLM port. Defaults to $VLLM_PORT or VLLM_DEFAULT_PORT.")
    args = p.parse_args()
    run(args.model, args.tp, args.mem_fraction_static, port=args.port)

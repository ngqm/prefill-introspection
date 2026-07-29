"""Stage 1 of the placebo-prefill pipeline: answer the benign prompts.

Generate one completion per prompt in data/benign_prompts.txt and write the
pairs to rq1_runs/benign_completions_<safe_mname>.jsonl, one row per prompt:
  {"prompt_idx": int, "prompt": str, "response": str, "model": str, "response_chars": int}

scripts/finetune/build_placebo_table.py later truncates each completion to the
character length of the matching adversarial prefix.

Usage:
  python -m scripts.generate.gen_benign_completions --model Qwen/Qwen3-8B
"""
import argparse
import json

from core.config import OUT_DIR, ROOT_DIR, VLLM_DEFAULT_PORT, resolve_model
from core.utils import set_seed
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager


def load_benign_prompts():
    """Return the non-empty, stripped lines of data/benign_prompts.txt."""
    p = ROOT_DIR / "data" / "benign_prompts.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def run(model_name, tp, gpu_memory_utilization, port=VLLM_DEFAULT_PORT):
    """Generate one benign completion per prompt and write them to OUT_DIR.

    Do nothing when the output file already exists and is non-empty.
    """
    set_seed()
    model_path, safe_mname = resolve_model(model_name, ablated=False)
    out = OUT_DIR / f"benign_completions_{safe_mname}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[*] {out} already populated; skipping.")
        return

    prompts = load_benign_prompts()
    print(f"[*] Loaded {len(prompts)} benign prompts.")

    with VLLMServerManager(model_path, port=port, tensor_parallel=tp,
                           gpu_memory_utilization=gpu_memory_utilization):
        client = VLLMClient(model_path, port=port)
        rows = []
        for i, prompt in enumerate(prompts):
            msgs = [{"role": "user", "content": prompt}]
            # Long enough to truncate to any adversarial-prefix length (at most ~250 chars).
            resp, _ = client.generate_chat(msgs, max_tokens=512)
            rows.append({
                "prompt_idx": i,
                "prompt": prompt,
                "response": resp,
                "model": model_name,
                "response_chars": len(resp),
            })
            print(f"  [{i+1}/{len(prompts)}] chars={len(resp)} | {prompt[:60]}...")

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[ok] wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--port", type=int, default=VLLM_DEFAULT_PORT)
    args = p.parse_args()
    run(args.model, args.tp, args.gpu_memory_utilization, port=args.port)

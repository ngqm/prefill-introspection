"""Measure |<h, r>| of the residual stream onto the refusal direction r (tab:proj_magnitudes).

Checks that r, extracted at the end of the user prompt, is still active at the
follow-up turn. Reads r and its (layer, pos) from
activations/<model>/optimal_direction.pt and averages the absolute projection
over 80 behavior-change prompts at four transcript points:
  first_ctrl  end of [user: prompt]
  first_pref  end of [user: prompt, asst: prefilled response]
  fup_ctrl    end of [user: prompt, asst: control response, user: probe]
  fup_pref    end of [user: prompt, asst: prefilled response, user: probe]

Writes rq1_runs/proj_magnitudes_<slug>.json.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m src.analysis.projection_magnitudes --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.config import ABLATED_MODELS, FOLLOWUPS, OUT_DIR
from src.analysis.data import DATASETS, load_judged
from src.analysis.filters import is_behavior_change

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]
N_SAMPLES = 80


def load_samples(slug, n=N_SAMPLES, seed=42):
    """Return n shuffled behavior-change prompts with their control and prefilled responses."""
    out = []
    for ds in DATASETS:
        ctrl = load_judged(slug, ds, "static", "control")
        adv = load_judged(slug, ds, "adv", "adv_prefill")
        for pid in ctrl.keys() & adv.keys():
            if is_behavior_change(ctrl[pid], adv[pid]):
                out.append({"prompt": ctrl[pid]["prompt_text"],
                            "ctrl_resp": ctrl[pid]["initial_response"],
                            "adv_resp": adv[pid]["initial_response"]})
    random.Random(seed).shuffle(out)
    return out[:n]


def find_blocks(model):
    """Return the model's list of transformer blocks."""
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.language_model.model.layers,
    ]
    for f in candidates:
        try:
            return f(model)
        except AttributeError:
            continue
    raise RuntimeError(f"Could not find transformer blocks in {type(model).__name__}")


def main():
    """Compute mean |proj_r(h)| under the four conditions and save the JSON."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    args = ap.parse_args()

    hf_id = args.model
    slug = hf_id.replace("/", "__")
    d = torch.load(ABLATED_MODELS[hf_id].parent / "optimal_direction.pt", weights_only=False)
    direction = d["direction"].to(torch.bfloat16).cuda()
    layer = int(d["layer"])
    pos = int(d["pos"])  # end-relative: 0 = last token
    print(f"[{args.model}] layer={layer} pos=-{pos+1}  direction d={direction.shape}")

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    samples = load_samples(slug)
    print(f"[{args.model}] {len(samples)} samples")

    captured = {}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    blocks = find_blocks(model)
    print(f"[*] found blocks: {type(blocks).__name__}, len={len(blocks)}")
    handle = blocks[layer].register_forward_hook(hook)

    def project(messages):
        text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            model(ids)
        h = captured["h"][0]
        v = h[h.shape[0] - 1 - pos].to(torch.bfloat16).to(direction.device)
        return abs((v @ direction).item())

    probe = FOLLOWUPS["F1_minimal"]
    results = {"first_ctrl": [], "first_pref": [], "fup_ctrl": [], "fup_pref": []}
    for i, s in enumerate(samples):
        user = {"role": "user", "content": s["prompt"]}
        ctrl = {"role": "assistant", "content": s["ctrl_resp"]}
        pref = {"role": "assistant", "content": s["adv_resp"]}
        followup = {"role": "user", "content": probe}
        try:
            results["first_ctrl"].append(project([user]))
            results["first_pref"].append(project([user, pref]))
            results["fup_ctrl"].append(project([user, ctrl, followup]))
            results["fup_pref"].append(project([user, pref, followup]))
        except Exception as e:
            print(f"[!] sample {i} failed: {e}")
            continue
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    handle.remove()
    print(f"\nResults for {args.model} (mean |proj|):")
    for k, vals in results.items():
        m = sum(vals)/len(vals) if vals else float('nan')
        print(f"  {k:10s}  mean={m:7.3f}  n={len(vals)}")

    out_path = OUT_DIR / f"proj_magnitudes_{slug}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "layer": layer, "pos": pos,
        "results": results,
        "means": {k: (sum(v)/len(v) if v else None) for k, v in results.items()},
        "n_samples": len(samples),
    }, indent=2))
    print(f"[ok] saved {out_path}")


if __name__ == "__main__":
    main()

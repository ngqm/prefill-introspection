"""Unified Modal generation entrypoint.

Runs the SAME experiment pipeline as scripts/generate/run_*.py, but the whole
job (vLLM server + generation loop) executes in one Modal container writing to a
Modal volume. One `@app.function` handles every condition, so this replaces the
per-model bespoke modal generation scripts.

Two ways to invoke it:

  # directly
  modal run scripts/modal/modal_generate.py --condition adv \
      --model meta-llama/Llama-3.1-8B-Instruct --dataset harmbench

  # transparently, via the --backend flag on a local run script
  python -m scripts.generate.run_adv --model ... --dataset ... --backend modal

Fetch results:
  modal volume get introspection-gen /out ./gen_out

Scope: base models only (an HF id is served directly). Ablated/local checkpoints
are not uploaded here; use the dedicated ablation-generation path for those.
Placebo prefills need a per-prompt table (build_placebo_table.py) and are run
locally; this entrypoint covers control / static / adv.
"""
import os
from pathlib import Path
import modal

APP_NAME = "introspection-gen"
VOLUME = "introspection-gen"
# Local repo root to mount into the image. Derived from this file's location so
# it works wherever the repo is cloned; override with $PROJECT_LOCAL if needed.
# Only used locally when the image is built; inside the Modal container this
# module lives at /root/, where parents[2] does not exist, so fall back safely.
_here = Path(__file__).resolve()
PROJECT_LOCAL = os.environ.get("PROJECT_LOCAL") or (
    str(_here.parents[2]) if len(_here.parents) > 2 else "/proj")

for _d in ("optimized_prefixes", "data"):
    os.makedirs(os.path.join(PROJECT_LOCAL, _d), exist_ok=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers==4.53.2", "accelerate==1.0.1",
                 "datasets==3.0.1", "vllm==0.10.0", "sentencepiece", "protobuf",
                 "huggingface_hub", "hf_transfer", "httpx", "psutil", "openai",
                 "pandas", "tqdm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(f"{PROJECT_LOCAL}/core", remote_path="/proj/core")
    .add_local_dir(f"{PROJECT_LOCAL}/experiments", remote_path="/proj/experiments")
    .add_local_dir(f"{PROJECT_LOCAL}/llm", remote_path="/proj/llm")
    .add_local_dir(f"{PROJECT_LOCAL}/optimized_prefixes", remote_path="/proj/optimized_prefixes")
    .add_local_dir(f"{PROJECT_LOCAL}/data", remote_path="/proj/data")
)
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
OUT_DIR = "/data/out"


@app.function(gpu="A100-80GB", volumes={"/data": vol},
              secrets=[modal.Secret.from_name("huggingface")],
              timeout=60 * 60 * 3, memory=64 * 1024)
def generate(model: str, dataset: str, condition: str = "adv",
             tp: int = 1, mem_fraction_static: float = 0.85, limit: int = 0):
    """Run one (model, dataset, condition) generation job on Modal.

    condition: control | static | adv
    limit>0 truncates the dataset (cheap smoke tests).
    """
    import sys, json, time
    from pathlib import Path
    sys.path.insert(0, "/proj"); os.chdir("/proj")
    os.environ["HF_HOME"] = "/data/hf"
    os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
    os.environ["VLLM_MAX_MODEL_LEN"] = "8192"

    from core.config import FOLLOWUPS, STATIC_PREFILLS, VLLM_DEFAULT_PORT
    from core.utils import set_seed, write_jsonl
    from core.dataset import DatasetLoader
    from llm.vllm_client import VLLMClient
    from llm.vllm_serve import VLLMServerManager
    from experiments.static_prefix import StaticPrefillExperiment
    from experiments.adv_prefix import AdvPrefillExperiment

    set_seed()
    safe = model.replace("/", "__")
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader()
    df = loader.load(dataset).to_dict("records")
    if limit:
        df = df[:limit]

    def save(results, prefix):
        suffix_map = {"F1_minimal": "_f1", "F2_tamper_check": "_f2",
                      "F3_minimal_truncated": "_f3", "F4_tamper_check_truncated": "_f4"}
        for fid, suf in suffix_map.items():
            path = out / f"{prefix}_{safe}_{dataset}{suf}.jsonl"
            if path.exists(): path.unlink()
            for rec in results:
                if "error" in rec:
                    write_jsonl(path, rec); continue
                base = {k: v for k, v in rec.items() if k != "followups"}
                if fid in rec.get("followups", {}):
                    write_jsonl(path, {**base, "followup_id": fid,
                                       "followup_text": FOLLOWUPS[fid],
                                       "followup_response": rec["followups"][fid]})
            n = sum(1 for _ in open(path)) if path.exists() else 0
            print(f"[{safe}][{dataset}] {prefix}{suf} n={n}", flush=True)

    t0 = time.time()
    print(f"=== serving {model} ({condition}) ===", flush=True)
    with VLLMServerManager(model, port=VLLM_DEFAULT_PORT, tensor_parallel=tp,
                           mem_fraction_static=mem_fraction_static):
        print(f"serve ready {time.time()-t0:.0f}s", flush=True)
        client = VLLMClient(model, port=VLLM_DEFAULT_PORT)
        if condition == "control":
            res = StaticPrefillExperiment(client, FOLLOWUPS, {}).run_sweep(df, concurrency=48)
            save(res, "gen")
        elif condition == "static":
            res = StaticPrefillExperiment(client, FOLLOWUPS, STATIC_PREFILLS).run_sweep(df, concurrency=48)
            save(res, "gen")
        elif condition == "adv":
            pf = Path(f"/proj/optimized_prefixes/optimized_prefixes_{safe}_{dataset}.json")
            if not pf.exists():
                raise FileNotFoundError(
                    f"{pf} not in the image: run scripts/generate/optimize_adv_prefixes.py "
                    f"locally first, or use condition=control|static.")
            prefixes = json.load(open(pf))
            res = AdvPrefillExperiment(client, FOLLOWUPS, prefixes).run_sweep(df, concurrency=48)
            save(res, "gen_adv")
        else:
            raise ValueError(f"condition must be control|static|adv, got {condition!r}")
        vol.commit()
    print(f"=== DONE {time.time()-t0:.0f}s ===", flush=True)
    return f"{safe}/{dataset}/{condition}"


def dispatch(condition: str, model: str, dataset: str, tp: int = 1,
             mem_fraction_static: float = 0.85, limit: int = 0, **_ignored):
    """Run one job on Modal from a local process (used by llm.backend.run_on_modal).

    Results land on the `introspection-gen` volume; fetch with
    `modal volume get introspection-gen /out ./gen_out`.
    """
    with app.run():
        tag = generate.remote(model=model, dataset=dataset, condition=condition,
                              tp=tp, mem_fraction_static=mem_fraction_static, limit=limit)
    print(f"[modal] finished {tag}. Fetch: modal volume get {VOLUME} /out ./gen_out")
    return tag


@app.local_entrypoint()
def main(model: str, dataset: str = "harmbench", condition: str = "adv",
         tp: int = 1, limit: int = 0):
    print(generate.remote(model=model, dataset=dataset, condition=condition,
                          tp=tp, limit=limit))

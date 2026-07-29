"""Run one generation job (vLLM server plus generation loop) in a Modal container.

Runs the same experiment as scripts/generate/run_generate.py and writes the
same gen[_adv]_<model>_<dataset>_f{1,2}.jsonl files to the Modal volume. Serves
base models only (an HF id); run ablated checkpoints with --backend local.
Placebo prefills need a per-prompt table and run locally.

Two ways to invoke it:

  # directly
  modal run scripts/modal/modal_generate.py --condition adv \\
      --model meta-llama/Llama-3.1-8B-Instruct --dataset harmbench

  # through the --backend flag of the local entrypoint
  python -m scripts.generate.run_generate --condition adv --model ... --dataset ... --backend modal

Fetch results:
  modal volume get introspection-gen /out ./gen_out
"""
import os
from pathlib import Path
import modal

APP_NAME = "introspection-gen"
VOLUME = "introspection-gen"
# Repo root mounted into the image, derived from this file's location (override
# with $PROJECT_LOCAL). Inside the container this module sits at /root/, where
# parents[2] does not exist.
_here = Path(__file__).resolve()
PROJECT_LOCAL = os.environ.get("PROJECT_LOCAL") or (
    str(_here.parents[2]) if len(_here.parents) > 2 else "/proj")

for _d in ("optimized_prefixes", "data"):
    os.makedirs(os.path.join(PROJECT_LOCAL, _d), exist_ok=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # fastapi and the instrumentator are pinned since a fresh vllm install
    # pulls a fastapi release under which every API request returns 500.
    .pip_install("torch==2.7.1", "transformers==4.53.2", "accelerate==1.0.1",
                 "datasets==3.0.1", "vllm==0.10.0", "fastapi==0.115.6",
                 "prometheus-fastapi-instrumentator==7.0.0", "sentencepiece", "protobuf",
                 "huggingface_hub", "hf_transfer", "httpx", "psutil", "openai",
                 "pandas", "tqdm", "python-dotenv")
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
             tp: int = 1, gpu_memory_utilization: float = 0.85, limit: int = 0):
    """Run one (model, dataset, condition) generation job on Modal.

    condition is control, static, or adv; limit > 0 truncates the dataset for
    smoke tests.
    """
    import sys, time
    sys.path.insert(0, "/proj"); os.chdir("/proj")
    os.environ["HF_HOME"] = "/data/hf"
    os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
    os.environ["VLLM_MAX_MODEL_LEN"] = "8192"

    from core.config import FOLLOWUPS, STATIC_PREFILLS, VLLM_CONCURRENCY, VLLM_DEFAULT_PORT
    from core.utils import set_seed
    from core.dataset import DatasetLoader
    from llm.vllm_client import VLLMClient
    from llm.vllm_serve import VLLMServerManager
    from experiments.adv_prefix import AdvPrefillExperiment, load_optimized_prefixes
    from experiments.prefill_experiment import save_results
    from experiments.static_prefix import StaticPrefillExperiment

    if condition not in ("control", "static", "adv"):
        raise ValueError(f"condition must be control|static|adv, got {condition!r}")
    set_seed()
    safe = model.replace("/", "__")
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    df = DatasetLoader().load(dataset).to_dict("records")
    if limit:
        df = df[:limit]
    prefixes = load_optimized_prefixes(model, dataset) if condition == "adv" else None

    t0 = time.time()
    print(f"=== serving {model} ({condition}) ===", flush=True)
    with VLLMServerManager(model, port=VLLM_DEFAULT_PORT, tensor_parallel=tp,
                           gpu_memory_utilization=gpu_memory_utilization):
        print(f"serve ready {time.time()-t0:.0f}s", flush=True)
        client = VLLMClient(model, port=VLLM_DEFAULT_PORT)
        if condition == "adv":
            exp, base_filename = AdvPrefillExperiment(client, FOLLOWUPS, prefixes), f"gen_adv_{safe}"
        else:
            prefills = STATIC_PREFILLS if condition == "static" else {}
            exp, base_filename = StaticPrefillExperiment(client, FOLLOWUPS, prefills), f"gen_{safe}"
        save_results(exp.run_sweep(df, concurrency=VLLM_CONCURRENCY), out, base_filename, dataset)
        vol.commit()
    print(f"=== DONE {time.time()-t0:.0f}s ===", flush=True)
    return f"{safe}/{dataset}/{condition}"


def dispatch(condition: str, model: str, dataset: str, tp: int = 1,
             gpu_memory_utilization: float = 0.85, limit: int = 0):
    """Run one job on Modal from a local process (used by run_generate --backend modal).

    Results land on the `introspection-gen` volume; fetch with
    `modal volume get introspection-gen /out ./gen_out`.
    """
    with app.run():
        tag = generate.remote(model=model, dataset=dataset, condition=condition,
                              tp=tp, gpu_memory_utilization=gpu_memory_utilization, limit=limit)
    print(f"[modal] finished {tag}. Fetch: modal volume get {VOLUME} /out ./gen_out")
    return tag


@app.local_entrypoint()
def main(model: str, dataset: str = "harmbench", condition: str = "adv",
         tp: int = 1, limit: int = 0):
    """Run one generation job on Modal from the command line."""
    print(generate.remote(model=model, dataset=dataset, condition=condition,
                          tp=tp, limit=limit))

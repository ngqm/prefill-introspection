"""Unified backend seam for the generation scripts.

Every `scripts/generate/run_*.py` does the same thing: build a client, wrap it in
an experiment, and call `exp.run_sweep(df)`. The only thing that varies is where
the model runs. This module centralizes that choice behind one `--backend` flag:

    local       spawn a local vLLM server, run the loop locally (default)
    openrouter  hosted models, run the loop locally (base models only)
    modal       run the whole job on Modal (see scripts/modal/modal_generate.py)

`sglang` is accepted as a legacy alias for `local`: the scripts historically said
`--backend sglang`, but the local server has been vLLM since the sglang env was
retired (llm/vllm_serve.py is a drop-in for the same OpenAI endpoint).

Usage in a run script:

    from llm.backend import client_for, is_modal, run_on_modal
    if is_modal(args.backend):
        run_on_modal(condition="adv", model=args.model, dataset=args.dataset, ...)
    else:
        with client_for(args.backend, model_path, model_name, tp=args.tp) as client:
            exp = AdvPrefillExperiment(client, FOLLOWUPS, prefixes)
            results = exp.run_sweep(df, ...)
        save_results(results, ...)
"""
import os
from contextlib import contextmanager
from typing import List, Optional

from core.config import VLLM_DEFAULT_PORT
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager
from llm.openrouter import OpenRouterClient

LOCAL_ALIASES = {"local", "vllm", "sglang"}
BACKEND_CHOICES = ["local", "modal", "openrouter", "sglang"]


def is_modal(backend: str) -> bool:
    return backend.lower() == "modal"


@contextmanager
def client_for(backend: str, model_path: str, model_name: str, *,
               port: int = VLLM_DEFAULT_PORT, tp: int = 1,
               mem_fraction_static: float = 0.80,
               extra_args: Optional[List[str]] = None, ablated: bool = False):
    """Yield a ready client for a *locally orchestrated* backend.

    - local (default): spawn a local vLLM server on `port` and yield a client to it.
    - openrouter: yield a hosted client (no local server; base models only).

    `modal` is not a local client; dispatch it with `run_on_modal()` instead.
    """
    b = backend.lower()
    if b in LOCAL_ALIASES:
        with VLLMServerManager(model_path, port=port, tensor_parallel=tp,
                               mem_fraction_static=mem_fraction_static,
                               extra_args=extra_args):
            # model_path may be an HF id or a local ablated-checkpoint dir; the
            # client loads the tokenizer from it and addresses the local server.
            yield VLLMClient(model_path, port=port)
    elif b == "openrouter":
        if ablated:
            raise ValueError("--backend openrouter cannot serve local ablated "
                             "checkpoints; use --backend local or --backend modal.")
        yield OpenRouterClient(model_name, os.getenv("OPENROUTER_API_KEY"))
    elif b == "modal":
        raise ValueError("--backend modal runs the whole job remotely; call "
                         "run_on_modal(...) rather than client_for(...).")
    else:
        raise ValueError(f"Unknown backend: {backend!r} "
                         f"(choose from {BACKEND_CHOICES}).")


def run_on_modal(condition: str, model: str, dataset: str, **kwargs):
    """Dispatch a generation job to Modal (whole job runs in a Modal container).

    Thin wrapper over the unified entrypoint in scripts/modal/modal_generate.py,
    imported lazily so `modal` stays an optional dependency for local runs.
    """
    from scripts.modal.modal_generate import dispatch  # lazy: modal is optional
    return dispatch(condition=condition, model=model, dataset=dataset, **kwargs)

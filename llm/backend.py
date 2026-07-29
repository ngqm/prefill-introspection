"""Build the LLM client for a locally orchestrated generation backend.

    local       spawn a local vLLM server and talk to it (default)
    openrouter  hosted base models through OpenRouter

The third backend, modal, runs the whole job remotely: scripts/generate/run_generate.py
dispatches it to scripts/modal/modal_generate.py.
"""
import os
from contextlib import contextmanager
from typing import List, Optional

from core.config import VLLM_DEFAULT_PORT
from llm.vllm_client import VLLMClient
from llm.vllm_serve import VLLMServerManager
from llm.openrouter import OpenRouterClient

BACKEND_CHOICES = ["local", "modal", "openrouter"]


@contextmanager
def client_for(backend: str, model_path: str, model_name: str, *,
               port: int = VLLM_DEFAULT_PORT, tp: int = 1,
               gpu_memory_utilization: float = 0.80,
               extra_args: Optional[List[str]] = None, ablated: bool = False):
    """Yield a ready client for the local or openrouter backend.

    model_path is what the local server loads (an HF id or a local checkpoint
    dir); model_name is the HF id OpenRouter addresses. Raise ValueError for
    modal, for an ablated model on openrouter, and for an unknown backend.
    """
    b = backend.lower()
    if b == "local":
        with VLLMServerManager(model_path, port=port, tensor_parallel=tp,
                               gpu_memory_utilization=gpu_memory_utilization,
                               extra_args=extra_args):
            yield VLLMClient(model_path, port=port)
    elif b == "openrouter":
        if ablated:
            raise ValueError("--backend openrouter cannot serve local ablated "
                             "checkpoints; use --backend local.")
        yield OpenRouterClient(model_name, os.getenv("OPENROUTER_API_KEY"))
    elif b == "modal":
        raise ValueError("--backend modal runs the whole job remotely; dispatch it "
                         "with scripts.modal.modal_generate.dispatch.")
    else:
        raise ValueError(f"Unknown backend: {backend!r} "
                         f"(choose from {BACKEND_CHOICES}).")

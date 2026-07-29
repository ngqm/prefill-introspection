"""Thin manager that boots a local vLLM OpenAI-compatible server.

The interface (constructor kwargs, context-manager protocol) dates from the
retired sglang-server era and is kept unchanged so callsites never churned.
llm.vllm_client.VLLMClient, an OpenAI client over a local http endpoint,
talks to this server via the standard /v1/completions and /v1/chat/completions
endpoints.
"""
import os
import sys
import time
import socket
import signal
import gc
import subprocess
import psutil
import httpx
from typing import List, Optional


class VLLMServerManager:
    """Context manager that launches `vllm serve` (OpenAI-compatible) and
    tears it down on exit. Keeps the sglang-era manager interface so callers
    only need to swap the class."""

    def __init__(self, model_path: str, port: int = 30000, tensor_parallel: int = 1,
                 mem_fraction_static: float = 0.80, extra_args: Optional[List[str]] = None):
        self.model_path = model_path
        self.port = port
        self.tensor_parallel = tensor_parallel
        # vLLM calls this --gpu-memory-utilization; the kwarg name is kept from
        # the sglang era for callsite compatibility.
        self.mem_fraction_static = mem_fraction_static
        self.extra_args = extra_args or []
        self.process = None

    def __enter__(self):
        self.ensure_port_free()
        log_path = os.environ.get("VLLM_LOG_PATH") or os.environ.get("SGLANG_LOG_PATH")
        print(f"[*] Launching vLLM: {self.model_path} on port {self.port} "
              f"(tp={self.tensor_parallel}, gpu-memory-utilization={self.mem_fraction_static:.2f}, "
              f"log={log_path or 'DEVNULL'})...")

        # Cap model context to fit KV cache on a single 24GB GPU after weights.
        # Llama-3.1-8B / Qwen3 default to 131K-32K context, which requires more
        # KV cache than we have once weights are loaded; the longest conversation
        # we generate (initial response 2048 + follow-up history ~600) sits well
        # under 8192. Override via VLLM_MAX_MODEL_LEN if you need more.
        max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN", "8192")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--gpu-memory-utilization", f"{self.mem_fraction_static:.2f}",
            "--max-model-len", max_model_len,
            # vLLM's tokenizer pulls chat_template / generation_config automatically.
        ]
        # Skip cudagraph capture (slow on large models, especially with TP>1).
        # Set VLLM_ENFORCE_EAGER=1 to add --enforce-eager.
        if os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
            cmd.append("--enforce-eager")
        # Bypass vLLM's custom all-reduce; falls back to standard NCCL all-reduce.
        # Workaround for TP>1 NCCL crashes with newer drivers / mismatched runtimes.
        if os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "").lower() in ("1", "true", "yes"):
            cmd.append("--disable-custom-all-reduce")
        # Quantization options (two distinct modes):
        # 1. Inflight NF4: VLLM_QUANTIZATION=bitsandbytes (no VLLM_LOAD_FORMAT).
        # 2. Pre-quantized bnb checkpoint: VLLM_QUANTIZATION=bitsandbytes +
        #    VLLM_LOAD_FORMAT=bitsandbytes. Use this for pre-saved int8 checkpoints
        #    whose config.json already has quantization_config.
        q = os.environ.get("VLLM_QUANTIZATION")
        if q:
            cmd += ["--quantization", q]
        lf = os.environ.get("VLLM_LOAD_FORMAT")
        if lf:
            cmd += ["--load-format", lf]
        # Cap concurrent sequences so vLLM doesn't reserve a huge KV cache at
        # startup. Default is 256; lowering reduces warmup time on TP>1.
        max_num_seqs = os.environ.get("VLLM_MAX_NUM_SEQS")
        if max_num_seqs:
            cmd += ["--max-num-seqs", str(max_num_seqs)]
        cmd += self.extra_args

        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            self._log_fh = open(log_path, "w")
            stdout_target, stderr_target = self._log_fh, subprocess.STDOUT
        else:
            self._log_fh = None
            stdout_target, stderr_target = subprocess.DEVNULL, subprocess.STDOUT

        self.process = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            start_new_session=True,
        )

        try:
            self.wait_for_server()
        except Exception:
            if log_path and os.path.exists(log_path):
                print(f"[!] vLLM launch failed; last 40 lines of {log_path}:")
                try:
                    with open(log_path) as f:
                        tail = f.readlines()[-40:]
                    print("".join(tail))
                except Exception:
                    pass
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[*] Shutting down vLLM on port {self.port}...")
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception:
                pass
            self.process.wait()
        if getattr(self, "_log_fh", None):
            try:
                self._log_fh.close()
            except Exception:
                pass

        self.ensure_port_free()
        self.clear_cuda_memory()

    def wait_for_server(self, timeout: int = None):
        if timeout is None:
            timeout = int(os.environ.get("VLLM_WAIT_TIMEOUT", 1200))
        start = time.time()
        last_err = None
        while time.time() - start < timeout:
            # Check process didn't die.
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server process exited with code {self.process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    resp = httpx.get(f"http://127.0.0.1:{self.port}/v1/models", timeout=2)
                    if resp.status_code == 200:
                        return True
            except Exception as e:
                last_err = e
                time.sleep(2)
        raise RuntimeError(f"vLLM failed to start on port {self.port} (last: {last_err!r})")

    def ensure_port_free(self):
        import getpass
        me = getpass.getuser()
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            try:
                conns = proc.net_connections(kind='inet')
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
            for conn in conns or []:
                try:
                    if getattr(conn, "laddr", None) and conn.laddr.port == self.port:
                        if proc.info.get('username') != me:
                            raise RuntimeError(
                                f"Port {self.port} is held by PID {proc.info['pid']} "
                                f"owned by {proc.info.get('username')!r}; refusing to kill "
                                f"another user's process. Set VLLM_DEFAULT_PORT to a free port.")
                        os.kill(proc.info['pid'], signal.SIGKILL)
                except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                    continue

    def clear_cuda_memory(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

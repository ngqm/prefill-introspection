"""Boot a local vLLM OpenAI-compatible server for VLLMClient to talk to."""
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
    """Context manager that launches a local vLLM OpenAI-compatible server.

    model_path is the HuggingFace id or checkpoint path; port, tensor_parallel,
    and gpu_memory_utilization map to the matching server flags, and
    extra_args are appended verbatim to the launch command.
    """

    def __init__(self, model_path: str, port: int = 30000, tensor_parallel: int = 1,
                 gpu_memory_utilization: float = 0.80, extra_args: Optional[List[str]] = None):
        self.model_path = model_path
        self.port = port
        self.tensor_parallel = tensor_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = extra_args or []
        self.process = None

    def __enter__(self):
        """Free the port, launch the vLLM subprocess, and block until it is ready.

        Optional env vars: VLLM_LOG_PATH (server log file), VLLM_MAX_MODEL_LEN
        (default 8192), VLLM_ENFORCE_EAGER and VLLM_DISABLE_CUSTOM_ALL_REDUCE
        (set to 1 to add the matching flag), and VLLM_QUANTIZATION,
        VLLM_LOAD_FORMAT, VLLM_MAX_NUM_SEQS (passed as --quantization,
        --load-format, --max-num-seqs).
        """
        self.ensure_port_free()
        log_path = os.environ.get("VLLM_LOG_PATH")
        print(f"[*] Launching vLLM: {self.model_path} on port {self.port} "
              f"(tp={self.tensor_parallel}, gpu-memory-utilization={self.gpu_memory_utilization:.2f}, "
              f"log={log_path or 'DEVNULL'})...")

        # The longest conversation (2048-token initial response plus follow-up
        # history) fits in 8192 tokens; a smaller context leaves room for KV cache.
        max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN", "8192")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--gpu-memory-utilization", f"{self.gpu_memory_utilization:.2f}",
            "--max-model-len", max_model_len,
        ]
        if os.environ.get("VLLM_ENFORCE_EAGER", "").lower() in ("1", "true", "yes"):
            cmd.append("--enforce-eager")
        if os.environ.get("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "").lower() in ("1", "true", "yes"):
            cmd.append("--disable-custom-all-reduce")
        q = os.environ.get("VLLM_QUANTIZATION")
        if q:
            cmd += ["--quantization", q]
        lf = os.environ.get("VLLM_LOAD_FORMAT")
        if lf:
            cmd += ["--load-format", lf]
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
        """Kill the server process group, free the port, and clear CUDA memory."""
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
        """Poll /v1/models until the server responds, then return True.

        timeout defaults to the VLLM_WAIT_TIMEOUT env var (1200 seconds).
        Raise RuntimeError if the process exits early or the timeout elapses.
        """
        if timeout is None:
            timeout = int(os.environ.get("VLLM_WAIT_TIMEOUT", 1200))
        start = time.time()
        last_err = None
        while time.time() - start < timeout:
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
        """Kill any of our own processes holding self.port.

        Raise RuntimeError if the port is held by another user's process
        instead of killing it.
        """
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
        """Run garbage collection and empty the CUDA cache if torch is available."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

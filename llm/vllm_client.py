"""OpenAI-compatible client for the local vLLM server.

VLLMClient renders chat templates, applies the Qwen `/no_think` soft switch,
strips the empty <think></think> block Qwen emits, and retries completions
against the server's /v1/completions endpoint.
"""
import os
import re
import time
import random
from typing import List, Dict, Tuple, Any
from openai import OpenAI
from transformers import AutoTokenizer
from llm.base import BaseLLMClient


class VLLMClient(BaseLLMClient):
    """Client for the local vLLM server at 127.0.0.1:port.

    Loads the tokenizer for model_name and authenticates with the VLLM_API_KEY
    env var when set.
    """

    def __init__(self, model_name: str, port: int = 30000):
        super().__init__(model_name)
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1",
                             api_key=os.getenv("VLLM_API_KEY", "EMPTY"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int, **kwargs) -> Tuple[str, Any]:
        """Render `messages` with the chat template and return (text, raw response).

        When the final turn is an assistant message, keep it open and continue
        the prefill text within the same turn. Otherwise add a generation prompt,
        append ` /no_think` to the rendered prompt (Qwen only), and strip the
        leading empty think block from the output.
        """
        if messages and messages[-1].get("role") == "assistant":
            # No /no_think suffix here: it would become part of the continuation,
            # and the template already renders an empty think block for this turn.
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True)
            return self.generate_text(prompt, max_tokens, apply_nothink=False, **kwargs)
        # Every generation in the paper uses this ` /no_think` path. The
        # `enable_thinking=False` template flag yields different greedy outputs
        # (Qwen3-32B control claim rate 81.8% against about 93%), so switching to
        # it would not reproduce the paper. Llama and Gemma are unaffected.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        text, resp = self.generate_text(prompt, max_tokens, **kwargs)
        return self._strip_think(text), resp

    # Leading empty think block the /no_think switch leaves in the output.
    _THINK_PREFIX_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)

    def _strip_think(self, text: str) -> str:
        """Remove the leading empty <think></think> block from text."""
        return self._THINK_PREFIX_RE.sub("", text or "")

    def generate_text(self, prompt: str, max_tokens: int, apply_nothink: bool = True, **kwargs) -> Tuple[str, Any]:
        """Complete `prompt` with retries and return (text, raw response).

        When apply_nothink is set, append the Qwen ` /no_think` switch first
        (a no-op for non-Qwen models). Retry up to six times with exponential
        backoff and raise RuntimeError if every attempt fails.
        """
        if apply_nothink:
            prompt = self.format_qwen_nothink(prompt)
        last_err = None
        for attempt in range(6):
            try:
                resp = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature", 0.0),
                    top_p=kwargs.get("top_p", 1.0),
                )
                return resp.choices[0].text, resp
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)

        raise RuntimeError(f"Generation failed: {repr(last_err)}")

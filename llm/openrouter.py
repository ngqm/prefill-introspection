import time
import random
import json
from typing import List, Dict, Tuple, Any
from openai import OpenAI, APIStatusError, RateLimitError, APIConnectionError
from llm.base import BaseLLMClient


class OpenRouterClient(BaseLLMClient):
    def __init__(self, model_name: str, api_key: str, max_retries: int = 6, request_timeout: float = 30.0):
        super().__init__(model_name)
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=request_timeout)
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.extra_body = {
            "reasoning": {"effort": "none", "exclude": True},
            "provider": {"require_parameters": False, "allow_fallbacks": True},
        }

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int, **kwargs) -> Tuple[str, Any]:
        # Apply Qwen parsing to user messages
        processed_messages = []
        for msg in messages:
            if msg["role"] == "user":
                msg = {"role": "user", "content": self.format_qwen_nothink(msg["content"])}
            processed_messages.append(msg)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=processed_messages,
                    max_tokens=max_tokens,
                    temperature=float(kwargs.get("temperature", 0.0)),
                    top_p=float(kwargs.get("top_p", 1.0)),
                    extra_body=self.extra_body,
                )
                text = resp.choices[0].message.content or ""
                return text, self._resp_to_jsonable(resp)
            
            except (RateLimitError, APIConnectionError, Exception) as e:
                last_err = e
                time.sleep(1.0 * (2 ** attempt) + random.random() * 0.25)
                
        raise RuntimeError(f"OpenRouter generation failed. Last error: {repr(last_err)}")

    def generate_text(self, prompt: str, max_tokens: int, **kwargs) -> Tuple[str, Any]:
        """OpenRouter relies primarily on chat completions; map text to a single user message."""
        return self.generate_chat([{"role": "user", "content": prompt}], max_tokens, **kwargs)

    def _resp_to_jsonable(self, resp) -> dict:
        if resp is None: return {}
        if hasattr(resp, "model_dump"):
            try: return resp.model_dump(mode="json")
            except TypeError: return resp.model_dump()
        if hasattr(resp, "to_dict"): return resp.to_dict()
        try: return json.loads(str(resp))
        except Exception: return {"_repr": repr(resp)}
"""Abstract LLM client interface shared by every backend.

BaseLLMClient fixes the two generation entrypoints (generate_chat,
generate_text) that the vLLM and OpenRouter clients implement, plus the
Qwen `/no_think` helper common to both.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class BaseLLMClient(ABC):
    """Abstract base for the project's LLM clients, keyed by model_name."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int, **kwargs) -> Tuple[str, Any]:
        """Generate a chat completion and return (text, raw response)."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int, **kwargs) -> Tuple[str, Any]:
        """Complete a raw text prompt and return (text, raw response).

        Used for strict prefilling, where the assistant turn stays open.
        """
        pass

    def format_qwen_nothink(self, text: str) -> str:
        """Append the ` /no_think` switch for Qwen models, once, if absent."""
        if "qwen" in self.model_name.lower() and not text.endswith(" /no_think"):
            return text + " /no_think"
        return text
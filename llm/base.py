from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List

class BaseLLMClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int, **kwargs) -> Tuple[str, Any]:
        """
        Generates a chat completion.
        Returns a tuple containing the generated text string and the raw response object.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int, **kwargs) -> Tuple[str, Any]:
        """
        Generates a completion from a raw text prompt (useful for strict prefilling).
        """
        pass

    def format_qwen_nothink(self, text: str) -> str:
        """Safely appends /no_think tag for Qwen models if required."""
        if "qwen" in self.model_name.lower() and not text.endswith(" /no_think"):
            return text + " /no_think"
        return text
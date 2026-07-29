"""Shared helpers for the reply judges and the RoBERTa reply classifiers.

victim_tokenizer loads the tokenizer of the model behind a gen_*.jsonl file, so
replies are truncated in that model's tokens. ReplyClassifier loads the F1 or
F2 classifier, from <probe>_classifier/ when you trained one locally and from
the Hugging Face Hub otherwise, and scores reply texts.
"""
import functools
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.config import ROOT_DIR

_FNAME_MODEL_RE = re.compile(
    r"gen_(?:adv_)?(?P<model>.+?)_(?:harmbench|socialharmbench|jailbreakbench|strongreject)_f[12]\.jsonl$"
)

# OpenRouter model ids are lowercase; the tokenizer lives under the Hugging Face repo id.
_HF_REPO_IDS = {
    "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
}


def victim_hf_id_from_filename(fname):
    """Parse 'gen_[adv_]MODEL_DATASET_fN.jsonl' into the victim's HF repo id, or None."""
    m = _FNAME_MODEL_RE.match(fname)
    if not m:
        return None
    slug = m.group("model").replace("__ablated", "")
    if "__" not in slug:
        return None
    hf_id = slug.replace("__", "/", 1)
    return _HF_REPO_IDS.get(hf_id.lower(), hf_id)


@functools.lru_cache(maxsize=None)
def get_victim_tokenizer(hf_id):
    """Load and cache a tokenizer; return None when it cannot be loaded."""
    try:
        return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] could not load tokenizer for {hf_id}: {type(e).__name__}: {str(e)[:120]}")
        return None


def victim_tokenizer(fname):
    """Return the victim tokenizer for a gen file name, or None."""
    hf_id = victim_hf_id_from_filename(fname)
    return get_victim_tokenizer(hf_id) if hf_id else None


HUB_CLASSIFIERS = {
    "f1": "minnq/prefill-introspection-intention-classifier",
    "f2": "minnq/prefill-introspection-tampering-classifier",
}


def classifier_dir(probe):
    """Return the directory where train_classifier saves the classifier for probe 'f1' or 'f2'."""
    return ROOT_DIR / f"{probe}_classifier" / "model_roberta-base"


def classifier_source(probe):
    """Return the local classifier directory if it exists, else the paper's Hub model id."""
    path = classifier_dir(probe)
    return str(path) if path.exists() else HUB_CLASSIFIERS[probe]


class ReplyClassifier:
    """Trained RoBERTa reply classifier for one probe.

    YES means the reply claims the output as intended (F1) or says it was
    tampered with (F2).
    """

    def __init__(self, probe):
        source = classifier_source(probe)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source).to(self.device).eval()
        # Checkpoints saved without id2label use class 1 for YES.
        self.yes_idx = self.model.config.label2id.get("YES", 1)

    def _logits(self, texts, batch_size=128, max_length=256, padding="max_length"):
        """Yield the logits for each batch of texts."""
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, padding=padding,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                yield self.model(**enc).logits

    def predict(self, texts, **kwargs):
        """Return the argmax label, "YES" or "NO", for each text."""
        return ["YES" if i == self.yes_idx else "NO"
                for logits in self._logits(texts, **kwargs)
                for i in logits.argmax(-1).cpu().tolist()]

    def p_yes(self, texts, **kwargs):
        """Return the softmax probability of YES for each text."""
        return [p for logits in self._logits(texts, **kwargs)
                for p in torch.softmax(logits, dim=-1)[:, self.yes_idx].cpu().tolist()]

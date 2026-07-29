import os
import json
import math
import random
from pathlib import Path
from typing import Dict, Any, Generator

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across standard, OS, and torch modules."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Safely try to seed torch if it's installed (used in generator scripts)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def make_json_safe(obj: Any) -> Any:
    """Recursively cleans objects to be JSON serializable (handles NaN/Inf floats)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj

def write_jsonl(path: Path | str, record: Dict[str, Any], mode: str = "a"):
    """Appends a single dictionary record to a JSONL file cleanly."""
    path = Path(path)
    # Ensure the parent directory exists just in case
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(make_json_safe(record), ensure_ascii=False) + "\n")

def read_jsonl(path: Path | str) -> Generator[Dict[str, Any], None, None]:
    """Generator that reads a JSONL file line by line without blowing up RAM."""
    path = Path(path)
    if not path.exists():
        return
        
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
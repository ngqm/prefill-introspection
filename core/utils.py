"""Small shared helpers for seeding and JSONL I/O.

set_seed for reproducible RNG state, make_json_safe for sanitizing NaN/Inf
before serialization, and write_jsonl / read_jsonl for appending and streaming
JSONL records.
"""
import os
import json
import math
import random
from pathlib import Path
from typing import Dict, Any, Generator

def set_seed(seed: int = 42):
    """Set the seed for reproducibility across standard, OS, and torch modules."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def make_json_safe(obj: Any) -> Any:
    """Recursively clean an object to be JSON serializable (map NaN/Inf floats to None)."""
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
    """Append a single dictionary record as one line to a JSONL file.

    Create the parent directory if needed. mode selects the file open mode
    (default "a" to append).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open(mode, encoding="utf-8") as f:
        f.write(json.dumps(make_json_safe(record), ensure_ascii=False) + "\n")

def read_jsonl(path: Path | str) -> Generator[Dict[str, Any], None, None]:
    """Yield records from a JSONL file line by line without loading it all into memory.

    Yield nothing if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        return
        
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
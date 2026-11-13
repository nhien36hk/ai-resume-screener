import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, List, Optional, Union


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def load_json(file_path: Union[str, Path], default_val: Any = None) -> Any:
    path = Path(file_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {} if default_val is None else default_val
    return {} if default_val is None else default_val


def save_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Timer:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time


def to_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    if not headers or not rows:
        return ""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
    separator_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines = [
        "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + data_lines)


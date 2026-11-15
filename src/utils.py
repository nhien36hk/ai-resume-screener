import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Union


def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def load_json(file_path: Union[str, Path], default_val: Any = None) -> Any:
    path = Path(file_path)
    if not path.exists():
        return {} if default_val is None else default_val
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise



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


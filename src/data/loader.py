import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.config import PROCESSED_DIR


def get_split_path(split_name: str, data_dir: Optional[Path] = None) -> Path:
    base_dir = Path(data_dir) if data_dir else PROCESSED_DIR
    return base_dir / f"{split_name}.jsonl"


def load_split_records(
    split_name: str,
    data_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    path = get_split_path(split_name, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file split tại: {path}. Vui lòng chạy scripts/03_merge_splits.py trước.")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split(
    split_name: str,
    data_dir: Optional[Path] = None,
    as_df: bool = True
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    records = load_split_records(split_name, data_dir)
    if not as_df:
        return records

    df = pd.DataFrame(records)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


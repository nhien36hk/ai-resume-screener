"""Nạp dữ liệu từ all.jsonl theo file index từng seed."""
from typing import Dict, List

import pandas as pd

from src.config import ALL_PAIRS_PATH, SPLITS_DIR
from src.utils import load_json


def load_all() -> pd.DataFrame:
    if not ALL_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file {ALL_PAIRS_PATH}. Vui lòng chạy scripts/03_merge_splits.py.")
    df = pd.read_json(ALL_PAIRS_PATH, lines=True)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


def split_ids(seed: int) -> Dict[str, List[str]]:
    path = SPLITS_DIR / f"seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file index {path}. Vui lòng chạy: python scripts/04_make_splits.py --seeds {seed}")
    return load_json(path)


def load_split(name: str, seed: int) -> pd.DataFrame:
    df_all = load_all()
    jds = set(split_ids(seed)[name])
    return df_all[df_all["job_id"].isin(jds)].reset_index(drop=True)

"""Nạp dữ liệu từ all.jsonl theo file index từng seed.

Hai cách chia (scheme):
- "jd": GroupShuffleSplit theo job_id; file data/splits/seed{N}.json lưu danh sách job_id.
- "strict": JD và CV đều chỉ thuộc một tập (theo góp ý GVHD); file data/splits/strict/seed{N}.json lưu danh sách pair_id.
"""
from typing import Dict, List

import pandas as pd

from src.config import ALL_PAIRS_PATH, SPLITS_DIR
from src.utils import load_json

SCHEMES = ("jd", "strict")


def load_all() -> pd.DataFrame:
    if not ALL_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file {ALL_PAIRS_PATH}. Vui lòng chạy scripts/03_merge_splits.py.")
    df = pd.read_json(ALL_PAIRS_PATH, lines=True)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df


def split_path(seed: int, scheme: str = "jd"):
    return SPLITS_DIR / f"seed{seed}.json" if scheme == "jd" else SPLITS_DIR / scheme / f"seed{seed}.json"


def split_ids(seed: int, scheme: str = "jd") -> Dict[str, List[str]]:
    path = split_path(seed, scheme)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file index {path}. Chạy: python scripts/04_make_splits.py --scheme {scheme} --seeds {seed}")
    return load_json(path)


def select_split(df: pd.DataFrame, name: str, seed: int, scheme: str = "jd") -> pd.DataFrame:
    """Cắt bảng (all.jsonl hoặc bảng feature) theo file index: scheme jd lọc job_id, strict lọc pair_id."""
    ids = set(split_ids(seed, scheme)[name])
    col = "job_id" if scheme == "jd" else "pair_id"
    missing = ids - set(df[col])
    if missing:
        raise ValueError(f"{len(missing)} {col} trong file index không còn trong dữ liệu (vd {sorted(missing)[:3]}). "
                         f"Chạy lại: python scripts/04_make_splits.py --scheme {scheme} --seeds {seed} --force")
    return df[df[col].isin(ids)].reset_index(drop=True)


def load_split(name: str, seed: int, scheme: str = "jd") -> pd.DataFrame:
    return select_split(load_all(), name, seed, scheme)

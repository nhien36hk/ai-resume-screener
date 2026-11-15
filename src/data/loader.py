import pandas as pd

from src.config import PROCESSED_DIR


def load_split(split_name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file split tại: {path}. Vui lòng chạy scripts/03_merge_splits.py trước.")

    df = pd.read_json(path, lines=True)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df

# Level 1: Thư viện chuẩn Python (Standard Library)
import os
from typing import Optional

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import pandas as pd

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "processed",
)


def load_split(
    split_name: str,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Nạp một tập dữ liệu (train, val hoặc test) từ tệp parquet."""
    dir_path = data_dir or DEFAULT_DATA_DIR
    parquet_path = os.path.join(dir_path, f"{split_name}.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    csv_path = os.path.join(dir_path, f"{split_name}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Không tìm thấy tệp {split_name}.parquet hoặc .csv tại {dir_path}")


def load_train_val_test(
    data_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nạp cả 3 tập train, val và test."""
    train_df = load_split("train", data_dir)
    val_df = load_split("val", data_dir)
    test_df = load_split("test", data_dir)
    return train_df, val_df, test_df


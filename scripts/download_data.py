import argparse
import sys
from pathlib import Path

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAIN_RATIO,
    LABEL_MAP,
    PROCESSED_DIR,
    RAW_DIR,
)


def download_and_combine(dataset_name: str) -> pd.DataFrame:
    """Tải dữ liệu từ Hugging Face và gộp train + test thành một DataFrame duy nhất."""
    ds = load_dataset(dataset_name)
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()
    return pd.concat([df_train, df_test], ignore_index=True)


def assign_job_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo mã định danh duy nhất (job_id) cho từng nội dung JD khác nhau."""
    unique_jds = {jd: f"JD_{i:04d}" for i, jd in enumerate(df["job_description_text"].unique())}
    df = df.copy()
    df["job_id"] = df["job_description_text"].map(unique_jds)
    df["label_text"] = df["label"]
    df["label"] = df["label_text"].map(LABEL_MAP)
    return df


def split_by_job_group(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df["job_id"]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    # 15% / (15% + 15%) = 0.50
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_df, groups=temp_df["job_id"]))

    val_df = temp_df.iloc[val_sub_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_sub_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def verify_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_jds = set(train_df["job_id"])
    val_jds = set(val_df["job_id"])
    test_jds = set(test_df["job_id"])

    assert len(train_jds & val_jds) == 0, "LỖI LEAKAGE: Trùng JD giữa Train và Val!"
    assert len(train_jds & test_jds) == 0, "LỖI LEAKAGE: Trùng JD giữa Train và Test!"
    assert len(val_jds & test_jds) == 0, "LỖI LEAKAGE: Trùng JD giữa Val và Test!"

    total_rows = len(train_df) + len(val_df) + len(test_df)
    print("=== KẾT QUẢ CHIA DỮ LIỆU BẢO ĐẢM ZERO DATA LEAKAGE ===")
    print(f"Tổng số mẫu: {total_rows}")
    print(f"Train: {len(train_df)} mẫu ({len(train_df)/total_rows:.1%}), {len(train_jds)} JDs")
    print(f"Val:   {len(val_df)} mẫu ({len(val_df)/total_rows:.1%}), {len(val_jds)} JDs")
    print(f"Test:  {len(test_df)} mẫu ({len(test_df)/total_rows:.1%}), {len(test_jds)} JDs")
    print("Xác nhận: 100% không có trùng lặp JD giữa bất kỳ tập nào!")


def save_datasets(
    df_all: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_all.to_parquet(raw_dir / "raw_dataset.parquet", index=False)
    df_all.to_csv(raw_dir / "raw_dataset.csv", index=False)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_parquet(processed_dir / f"{name}.parquet", index=False)
        split_df.to_csv(processed_dir / f"{name}.csv", index=False)

    print(f"Đã lưu thành công vào: {raw_dir} và {processed_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tải và chia dữ liệu CV-JD theo nhóm job_id.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME, help="Tên dataset trên Hugging Face.")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Tỷ lệ tập Train.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("1. Tải và gộp dataset từ HuggingFace...")
    df_all = download_and_combine(args.dataset)

    print("2. Gán job_id định danh và chuẩn hóa nhãn...")
    df_all = assign_job_ids(df_all)

    print("3. Phân chia GroupShuffleSplit theo job_id (70% - 15% - 15%)...")
    train_df, val_df, test_df = split_by_job_group(
        df_all,
        train_ratio=args.train_ratio,
        random_state=args.seed,
    )

    print("4. Xác thực rò rỉ dữ liệu...")
    verify_splits(train_df, val_df, test_df)

    print("5. Lưu trữ tệp dữ liệu vào thư mục data/...")
    save_datasets(df_all, train_df, val_df, test_df, RAW_DIR, PROCESSED_DIR)


if __name__ == "__main__":
    main()

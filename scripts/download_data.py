# Level 1: Thư viện chuẩn Python (Standard Library)
import os
import sys
from collections import Counter

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

# Level 3: Module nội bộ trong dự án (Project-Local Imports)
# (Chưa có module nội bộ cho bước data preparation)

DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}


def download_and_combine(dataset_name: str) -> pd.DataFrame:
    """Tải dữ liệu từ Hugging Face và gộp train + test thành một DataFrame duy nhất."""
    ds = load_dataset(dataset_name)
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    return df_all


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
    train_ratio: float = 0.70,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu thành Train / Val / Test theo nhóm job_id để triệt tiêu Data Leakage."""
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
    """Kiểm tra tính toàn vẹn và đảm bảo 0 rò rỉ dữ liệu giữa các tập."""
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
    raw_dir: str,
    processed_dir: str,
) -> None:
    """Lưu dữ liệu dưới cả định dạng Parquet và CSV."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Lưu tập raw
    df_all.to_parquet(os.path.join(raw_dir, "raw_dataset.parquet"), index=False)
    df_all.to_csv(os.path.join(raw_dir, "raw_dataset.csv"), index=False)

    # Lưu tập processed
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df.to_parquet(os.path.join(processed_dir, f"{name}.parquet"), index=False)
        split_df.to_csv(os.path.join(processed_dir, f"{name}.csv"), index=False)

    print(f"Đã lưu thành công vào: {raw_dir} và {processed_dir}")


def main():
    print("1. Tải và gộp dataset từ HuggingFace...")
    df_all = download_and_combine(DATASET_NAME)

    print("2. Gán job_id định danh và chuẩn hóa nhãn...")
    df_all = assign_job_ids(df_all)

    print("3. Phân chia GroupShuffleSplit theo job_id (70% - 15% - 15%)...")
    train_df, val_df, test_df = split_by_job_group(
        df_all,
        train_ratio=TRAIN_RATIO,
        random_state=RANDOM_STATE,
    )

    print("4. Xác thực rò rỉ dữ liệu...")
    verify_splits(train_df, val_df, test_df)

    print("5. Lưu trữ tệp dữ liệu vào thư mục data/...")
    save_datasets(df_all, train_df, val_df, test_df, RAW_DIR, PROCESSED_DIR)


if __name__ == "__main__":
    main()


import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAIN_RATIO,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.data.splitter import (
    assign_job_ids,
    download_and_combine,
    save_datasets,
    split_by_job_group,
    verify_splits,
)


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

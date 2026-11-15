"""Tải dataset gốc từ HuggingFace, gán job_id và lưu raw_dataset.parquet."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_NAME, RAW_DIR
from src.data.splitter import assign_job_ids, download_and_combine


def main():
    parser = argparse.ArgumentParser(description="Tải dataset gốc từ Hugging Face.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    print("1. Tải và gộp dataset từ HuggingFace...")
    df = download_and_combine(args.dataset)

    print("2. Gán job_id định danh và chuẩn hóa nhãn...")
    df = assign_job_ids(df)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "raw_dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Đã lưu thành công: {out_path} ({len(df)} dòng, {df['job_id'].nunique()} JDs)")


if __name__ == "__main__":
    main()

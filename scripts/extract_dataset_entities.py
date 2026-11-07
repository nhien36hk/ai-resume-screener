# Level 1: Thư viện chuẩn Python (Standard Library)
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional

# Đảm bảo root path nằm trong sys.path trước khi nạp module nội bộ
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import pandas as pd
from tqdm import tqdm

# Level 3: Module nội bộ trong dự án (Project-Local Imports)
from src.data.extractor import extract_jd_entities, extract_cv_entities


DEFAULT_INPUT_PATH = "data/processed/train.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/extracted_sample_25.json"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_SEED = 42
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def extract_with_retry(extract_fn, text: str, model_name: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """Hàm wrapper bọc gọi hàm trích xuất có cơ chế tự động thử lại khi gặp lỗi mạng."""
    for attempt in range(1, max_retries + 1):
        try:
            res = extract_fn(text, model_name=model_name)
            if res and isinstance(res, dict):
                return res
        except Exception as e:
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * attempt)
            else:
                print(f"\n[Warning] Lỗi trích xuất sau {max_retries} lần thử: {e}")
    return {}


def run_extraction(
    input_path: str,
    output_path: str,
    sample_size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    model_name: str = DEFAULT_MODEL,
    checkpoint_every: int = 5
) -> None:
    """Hàm điều phối luồng bóc tách dữ liệu có cache JD và lưu checkpoint liên tục."""
    print("=" * 70)
    print("HỆ THỐNG BÓC TÁCH THỰC THỂ CV & JD BẰNG AI (NEMOTRON 120B)")
    print(f"File nguồn: {input_path}")
    print(f"Mô hình sử dụng: {model_name}")
    print("=" * 70)

    # 1. Nạp dữ liệu
    df = pd.read_parquet(input_path)
    print(f"Tổng số mẫu trong tập dữ liệu: {len(df):,}")

    if sample_size and sample_size < len(df):
        random.seed(seed)
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f">>> Đã chọn ngẫu nhiên {sample_size} mẫu để kiểm thử (Seed={seed}).")
    else:
        df = df.reset_index(drop=True)

    # 2. Quản lý Checkpoint & JD Cache
    records: List[Dict[str, Any]] = []
    jd_cache: Dict[str, Dict[str, Any]] = {}
    completed_indices = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                records = saved_data.get("records", [])
                jd_cache = saved_data.get("jd_cache", {})
                completed_indices = {r["sample_idx"] for r in records}
                print(f">>> Tìm thấy checkpoint! Đã hoàn thành {len(records)} mẫu trước đó. Tiếp tục chạy...")
        except Exception as e:
            print(f">>> Không thể nạp checkpoint ({e}), bắt đầu chạy mới.")

    # 3. Vòng lặp bóc tách chính
    print(f"\nBắt đầu xử lý {len(df)} mẫu...")
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tiến trình bóc tách"):
        if idx in completed_indices:
            continue

        job_id = row["job_id"]
        jd_text = row["job_description_text"]
        cv_text = row["resume_text"]
        label = int(row["label"])
        label_text = row["label_text"]

        # 3.1 Bóc tách JD (Có Cache theo job_id)
        if job_id not in jd_cache:
            jd_entities = extract_with_retry(extract_jd_entities, jd_text, model_name)
            jd_cache[job_id] = jd_entities
        else:
            jd_entities = jd_cache[job_id]

        # 3.2 Bóc tách CV
        cv_entities = extract_with_retry(extract_cv_entities, cv_text, model_name)

        # 3.3 Lưu trữ record
        record = {
            "sample_idx": idx,
            "job_id": job_id,
            "label": label,
            "label_text": label_text,
            "jd_entities": jd_entities,
            "cv_entities": cv_entities,
            "jd_text_preview": jd_text[:300] + "..." if len(jd_text) > 300 else jd_text,
            "cv_text_preview": cv_text[:300] + "..." if len(cv_text) > 300 else cv_text,
        }
        records.append(record)
        completed_indices.add(idx)

        # 3.4 Lưu Checkpoint định kỳ
        if len(records) % checkpoint_every == 0 or len(records) == len(df):
            checkpoint_payload = {
                "total_completed": len(records),
                "model": model_name,
                "jd_cache": jd_cache,
                "records": records
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_payload, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("HOÀN THÀNH BÓC TÁCH THÀNH CÔNG!")
    print(f"- Tổng số mẫu đã xử lý: {len(records)}")
    print(f"- Số JD duy nhất được bóc tách: {len(jd_cache)}")
    print(f"- Tổng thời gian: {elapsed:.2f}s (Trung bình: {elapsed / max(len(records), 1):.2f}s/mẫu)")
    print(f"- File lưu kết quả: {output_path}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Kịch bản bóc tách thực thể CV/JD bằng AI Nemotron 120B.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, help="Đường dẫn file parquet đầu vào.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Đường dẫn file JSON đầu ra.")
    parser.add_argument("--sample", type=int, default=25, help="Số lượng mẫu ngẫu nhiên để test (mặc định: 25).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed chọn mẫu ngẫu nhiên.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Tên model NVIDIA.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Số mẫu mỗi lần lưu checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample,
        seed=args.seed,
        model_name=args.model,
        checkpoint_every=args.checkpoint_every
    )

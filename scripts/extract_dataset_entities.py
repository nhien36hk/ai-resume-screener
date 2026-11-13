# Level 1: Thư viện chuẩn Python (Standard Library)
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Đảm bảo root path nằm trong sys.path trước khi nạp module nội bộ
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import pandas as pd
from tqdm import tqdm

# Level 3: Module nội bộ trong dự án (Project-Local Imports)
from src.data.extractor import (
    DEFAULT_NVIDIA_MODEL,
    extract_jd_entities,
    extract_cv_entities
)


DEFAULT_INPUT_PATH = "data/processed/train.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/extracted_dataset.json"
DEFAULT_JD_CACHE_PATH = "data/processed/jd_cache.json"
DEFAULT_CV_CACHE_PATH = "data/processed/cv_cache.json"
DEFAULT_FAILED_PATH = "data/processed/failed_extractions.json"
DEFAULT_MODEL = DEFAULT_NVIDIA_MODEL
DEFAULT_SEED = 42
DEFAULT_CHECKPOINT_EVERY = 5
RATE_LIMIT_DELAY = 1.0  # NVIDIA xử lý 1.0s mượt mà


def compute_text_hash(text: str) -> str:
    """Tạo mã băm MD5 duy nhất cho một văn bản."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def load_json_file(file_path: str, default_val: Any) -> Any:
    """Nạp file JSON an toàn, trả về giá trị mặc định nếu file chưa tồn tại."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_val
    return default_val


def save_json_file(file_path: str, data: Any) -> None:
    """Lưu dữ liệu sang file JSON với encoding UTF-8 chuẩn xác."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_unique_jds(
    unique_jds: List[Tuple[str, str]],
    jd_cache: Dict[str, Any],
    failed_extractions: Dict[str, Any],
    model_name: str,
    jd_cache_path: str,
    failed_path: str,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
) -> Tuple[int, int]:
    """Bóc tách thực thể cho toàn bộ các JD duy nhất chưa có trong Cache."""
    needed_jds = [(j_id, text) for j_id, text in unique_jds if str(j_id) not in jd_cache]
    print(f"\n[PHASE 1: XỬ LÝ JOB DESCRIPTIONS] Cần bóc tách: {len(needed_jds)} / {len(unique_jds)} JD.")

    success_count = 0
    fail_count = 0

    for idx, (job_id, jd_text) in enumerate(tqdm(needed_jds, desc="Trích xuất JD"), 1):
        str_job_id = str(job_id)
        try:
            entities = extract_jd_entities(jd_text, model_name=model_name)
            jd_cache[str_job_id] = entities
            success_count += 1
            if str_job_id in failed_extractions.get("jds", {}):
                del failed_extractions["jds"][str_job_id]
        except Exception as e:
            fail_count += 1
            if "jds" not in failed_extractions:
                failed_extractions["jds"] = {}
            failed_extractions["jds"][str_job_id] = {
                "error": str(e),
                "timestamp": time.time(),
                "preview": jd_text[:300]
            }

        # Nghỉ nhẹ để bảo đảm tốc độ dưới trần TPM của Groq
        time.sleep(RATE_LIMIT_DELAY)

        if idx % checkpoint_every == 0 or idx == len(needed_jds):
            save_json_file(jd_cache_path, jd_cache)
            save_json_file(failed_path, failed_extractions)

    return success_count, fail_count


def extract_unique_cvs(
    unique_cvs: List[str],
    cv_cache: Dict[str, Any],
    failed_extractions: Dict[str, Any],
    model_name: str,
    cv_cache_path: str,
    failed_path: str,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
) -> Tuple[int, int]:
    """Bóc tách thực thể cho toàn bộ các CV duy nhất chưa có trong Cache."""
    needed_cvs = []
    for cv_text in unique_cvs:
        c_hash = compute_text_hash(cv_text)
        if c_hash not in cv_cache:
            needed_cvs.append((c_hash, cv_text))

    print(f"\n[PHASE 2: XỬ LÝ RESUMES / CV] Cần bóc tách: {len(needed_cvs)} / {len(unique_cvs)} CV.")

    success_count = 0
    fail_count = 0

    for idx, (c_hash, cv_text) in enumerate(tqdm(needed_cvs, desc="Trích xuất CV"), 1):
        t0 = time.time()
        try:
            entities = extract_cv_entities(cv_text, model_name=model_name)
            cv_cache[c_hash] = entities
            success_count += 1
            t_cv = time.time() - t0
            print(f" -> CV {idx}/{len(needed_cvs)} ({c_hash[:8]}): {t_cv:.2f}s | {len(entities.get('skills', []))} skills, {entities.get('total_years_experience', 0)} năm kn")
            if c_hash in failed_extractions.get("cvs", {}):
                del failed_extractions["cvs"][c_hash]
        except Exception as e:
            fail_count += 1
            t_cv = time.time() - t0
            print(f" -> [LỖI] CV {idx}/{len(needed_cvs)} ({c_hash[:8]}): Thất bại sau {t_cv:.2f}s: {e}")
            if "cvs" not in failed_extractions:
                failed_extractions["cvs"] = {}
            failed_extractions["cvs"][c_hash] = {
                "error": str(e),
                "timestamp": time.time(),
                "preview": cv_text[:300]
            }

        # Nghỉ nhẹ để duy trì kết nối ổn định
        time.sleep(RATE_LIMIT_DELAY)

        if idx % checkpoint_every == 0 or idx == len(needed_cvs):
            save_json_file(cv_cache_path, cv_cache)
            save_json_file(failed_path, failed_extractions)

    return success_count, fail_count


def run_pipeline(
    input_path: str,
    output_path: str,
    jd_cache_path: str = DEFAULT_JD_CACHE_PATH,
    cv_cache_path: str = DEFAULT_CV_CACHE_PATH,
    failed_path: str = DEFAULT_FAILED_PATH,
    sample_size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    model_name: str = DEFAULT_MODEL,
    all_splits: bool = False
) -> None:
    """Hàm điều phối luồng bóc tách dữ liệu kép (Double Cache) và ghép nối dataset."""
    print("=" * 75)
    print("HỆ THỐNG BÓC TÁCH THỰC THỂ CV & JD BẰNG AI (NVIDIA NEMOTRON 120B)")
    print(f"Mô hình sử dụng : {model_name}")
    print(f"File nguồn dữ liệu: {input_path}")
    print(f"Bộ nhớ Cache JD : {jd_cache_path}")
    print(f"Bộ nhớ Cache CV : {cv_cache_path}")
    print(f"File lưu lỗi riêng: {failed_path}")
    print("=" * 75)

    # 1. Nạp dữ liệu
    if all_splits:
        dir_name = os.path.dirname(input_path)
        splits = []
        for s_name in ["train.parquet", "val.parquet", "test.parquet"]:
            s_path = os.path.join(dir_name, s_name)
            if os.path.exists(s_path):
                splits.append(pd.read_parquet(s_path))
        df = pd.concat(splits, ignore_index=True) if splits else pd.read_parquet(input_path)
        print(f">>> Đã nạp gộp toàn bộ 3 tập train, val, test: {len(df):,} cặp mẫu.")
    else:
        df = pd.read_parquet(input_path)
        print(f">>> Đã nạp tập dữ liệu: {len(df):,} cặp mẫu.")

    # Nếu có chế độ sample để kiểm thử nhanh
    if sample_size and sample_size < len(df):
        random.seed(seed)
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f">>> [CHẾ ĐỘ TEST] Đang lấy ngẫu nhiên {sample_size} cặp mẫu để kiểm chứng (Seed={seed}).")
    else:
        df = df.reset_index(drop=True)

    # 2. Nạp Cache hiện có
    jd_cache: Dict[str, Any] = load_json_file(jd_cache_path, {})
    cv_cache: Dict[str, Any] = load_json_file(cv_cache_path, {})
    failed_extractions: Dict[str, Any] = load_json_file(failed_path, {"jds": {}, "cvs": {}})

    print(f">>> Trạng thái Cache: Đã có {len(jd_cache)} JDs và {len(cv_cache)} CVs được bóc tách.")
    print(f">>> Trạng thái Lỗi : {len(failed_extractions.get('jds', {}))} JDs lỗi, {len(failed_extractions.get('cvs', {}))} CVs lỗi.")

    # 3. Gom danh sách duy nhất
    unique_jds_df = df[["job_id", "job_description_text"]].drop_duplicates(subset=["job_id"])
    unique_jds = [(str(row["job_id"]), row["job_description_text"]) for _, row in unique_jds_df.iterrows()]
    unique_cvs = list(df["resume_text"].drop_duplicates())

    start_time = time.time()

    # 4. Phase 1: Bóc tách JD duy nhất
    jd_succ, jd_fail = extract_unique_jds(
        unique_jds, jd_cache, failed_extractions, model_name, jd_cache_path, failed_path
    )

    # 5. Phase 2: Bóc tách CV duy nhất
    cv_succ, cv_fail = extract_unique_cvs(
        unique_cvs, cv_cache, failed_extractions, model_name, cv_cache_path, failed_path
    )

    # 6. Phase 3: Ánh xạ kết quả ghép thành dataset hoàn chỉnh
    print(f"\n[PHASE 3: ÁNH XẠ KẾT QUẢ VÀO {len(df)} BẢN GHI DATASET]")
    dataset_records: List[Dict[str, Any]] = []
    missing_count = 0

    for idx, row in df.iterrows():
        job_id_str = str(row["job_id"])
        c_hash = compute_text_hash(row["resume_text"])

        extracted_jd = jd_cache.get(job_id_str)
        extracted_cv = cv_cache.get(c_hash)

        is_complete = bool(extracted_jd and extracted_cv)
        if not is_complete:
            missing_count += 1

        record = {
            "sample_idx": idx,
            "job_id": row["job_id"],
            "label": int(row["label"]),
            "label_text": row["label_text"],
            "jd_entities": extracted_jd,
            "cv_entities": extracted_cv,
            "is_complete": is_complete
        }
        dataset_records.append(record)

    # 7. Lưu file kết quả dataset hoàn chỉnh
    output_payload = {
        "total_records": len(dataset_records),
        "complete_records": len(dataset_records) - missing_count,
        "missing_records": missing_count,
        "unique_jds_total": len(unique_jds),
        "unique_cvs_total": len(unique_cvs),
        "model_used": model_name,
        "records": dataset_records
    }
    save_json_file(output_path, output_payload)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("BÁO CÁO TỔNG KẾT TIẾN TRÌNH:")
    print(f"- Tổng thời gian xử lý: {elapsed:.2f}s")
    print(f"- Tổng số JD duy nhất trong cache : {len(jd_cache)} (Mới bóc tách: {jd_succ}, Thất bại: {jd_fail})")
    print(f"- Tổng số CV duy nhất trong cache : {len(cv_cache)} (Mới bóc tách: {cv_succ}, Thất bại: {cv_fail})")
    print(f"- Số bản ghi hoàn thiện 100%      : {len(dataset_records) - missing_count} / {len(dataset_records)}")
    print(f"- File Dataset hoàn chỉnh         : {output_path}")
    print(f"- File Ghi nhận Lỗi riêng         : {failed_path}")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description="Hệ thống bóc tách thực thể CV/JD bằng AI (NVIDIA Nemotron 120B).")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, help="Đường dẫn file parquet đầu vào.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Đường dẫn file JSON đầu ra.")
    parser.add_argument("--jd-cache", type=str, default=DEFAULT_JD_CACHE_PATH, help="Đường dẫn lưu cache JD.")
    parser.add_argument("--cv-cache", type=str, default=DEFAULT_CV_CACHE_PATH, help="Đường dẫn lưu cache CV.")
    parser.add_argument("--failed-file", type=str, default=DEFAULT_FAILED_PATH, help="Đường dẫn lưu các mẫu lỗi.")
    parser.add_argument("--sample", type=int, default=None, help="Số lượng mẫu test (để None nếu chạy full).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed chọn mẫu ngẫu nhiên.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Tên model NVIDIA hoặc Groq.")
    parser.add_argument("--all-splits", action="store_true", help="Gộp cả 3 tập train, val, test để bóc tách toàn diện.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        jd_cache_path=args.jd_cache,
        cv_cache_path=args.cv_cache,
        failed_path=args.failed_file,
        sample_size=args.sample,
        seed=args.seed,
        model_name=args.model,
        all_splits=args.all_splits
    )

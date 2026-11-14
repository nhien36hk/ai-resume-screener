import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CV_CACHE_PATH,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_VLLM_MODEL,
    DEGREE_LEVELS,
    FAILED_EXTRACTIONS_PATH,
    JD_CACHE_PATH,
    MAJOR_GROUPS,
    PROCESSED_DIR,
)
from src.data.extractor import (
    BaseExtractorBackend,
    NvidiaAPIBackend,
    VLLMBackend,
    extract_cv_entities,
    extract_jd_entities,
)
from src.data.schemas import (
    CVEntities,
    JDEntities,
    SCHEMA_VERSION,
)
from src.utils import compute_text_hash, load_json, save_json, to_markdown_table


def needs_extraction(entry: Optional[Dict[str, Any]], schema_cls: Any) -> bool:
    if not entry or not isinstance(entry, dict):
        return True
    if entry.get("_schema_version", 1) < SCHEMA_VERSION:
        return True
    try:
        schema_cls.model_validate(entry)
        return False
    except Exception:
        return True


def backup_cache_file(cache_path: Path) -> None:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        backup_path = cache_path.with_suffix(".bak.json")
        if not backup_path.exists():
            shutil.copy2(cache_path, backup_path)


def load_dataset_splits(data_dir: Path) -> pd.DataFrame:
    splits = []
    for split_file in ["train.parquet", "val.parquet", "test.parquet"]:
        p = data_dir / split_file
        if p.exists():
            splits.append(pd.read_parquet(p))
    if not splits:
        raise FileNotFoundError(f"Không tìm thấy file parquet trong {data_dir}")
    return pd.concat(splits, ignore_index=True)


def process_jd_item(
    job_id: str,
    jd_text: str,
    backend: BaseExtractorBackend,
    model_name: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        res = extract_jd_entities(jd_text, backend=backend, model_name=model_name)
        return job_id, res, None
    except Exception as e:
        return job_id, None, str(e)


def process_cv_item(
    c_hash: str,
    cv_text: str,
    backend: BaseExtractorBackend,
    model_name: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        res = extract_cv_entities(cv_text, backend=backend, model_name=model_name)
        return c_hash, res, None
    except Exception as e:
        return c_hash, None, str(e)


def run_pipeline(args: argparse.Namespace) -> None:
    df = load_dataset_splits(args.data_dir)
    print(f">>> Đã nạp dữ liệu gộp cả 3 split: {len(df):,} cặp (CV, JD).")

    unique_jds_df = df[["job_id", "job_description_text"]].drop_duplicates(subset=["job_id"])
    unique_jds = [(str(row["job_id"]), row["job_description_text"]) for _, row in unique_jds_df.iterrows()]
    unique_cvs = list(df["resume_text"].drop_duplicates())

    jd_cache: Dict[str, Any] = load_json(args.jd_cache, {})
    cv_cache: Dict[str, Any] = load_json(args.cv_cache, {})
    failed_extractions: Dict[str, Any] = load_json(args.failed_file, {"jds": {}, "cvs": {}})
    failed_jds = failed_extractions.get("jds", {})
    failed_cvs = failed_extractions.get("cvs", {})

    needed_jds: List[Tuple[str, str]] = []
    if args.only in ("all", "jd"):
        for j_id, text in unique_jds:
            if args.force:
                needed_jds.append((j_id, text))
            elif args.retry_failed:
                if j_id in failed_jds:
                    needed_jds.append((j_id, text))
            elif needs_extraction(jd_cache.get(j_id), JDEntities):
                needed_jds.append((j_id, text))

    needed_cvs: List[Tuple[str, str]] = []
    if args.only in ("all", "cv"):
        for cv_text in unique_cvs:
            c_hash = compute_text_hash(cv_text)
            if args.force:
                needed_cvs.append((c_hash, cv_text))
            elif args.retry_failed:
                if c_hash in failed_cvs:
                    needed_cvs.append((c_hash, cv_text))
            elif needs_extraction(cv_cache.get(c_hash), CVEntities):
                needed_cvs.append((c_hash, cv_text))

    if args.sample is not None and args.sample > 0:
        needed_jds = needed_jds[:args.sample]
        failed_needed = [item for item in needed_cvs if item[0] in failed_cvs]
        normal_needed = [item for item in needed_cvs if item[0] not in failed_cvs]
        if failed_needed:
            needed_cvs = [failed_needed[0]] + normal_needed[:max(0, args.sample - 1)]
        else:
            needed_cvs = needed_cvs[:args.sample]

    print("\n" + "=" * 75)
    print("KẾ HOẠCH BÓC TÁCH THỰC THỂ (EXTRACTION PLAN):")
    print(f"- Backend              : {args.backend}")
    print(f"- Model                : {args.model}")
    print(f"- Tổng JD trong dữ liệu: {len(unique_jds)} (Cần xử lý: {len(needed_jds)})")
    print(f"- Tổng CV trong dữ liệu: {len(unique_cvs)} (Cần xử lý: {len(needed_cvs)})")
    print(f"- Đang có trong cache  : {len(jd_cache)} JDs, {len(cv_cache)} CVs")
    print(f"- Số mục từng lỗi trước: {len(failed_jds)} JDs, {len(failed_cvs)} CVs")
    print("=" * 75)

    if args.dry_run:
        print(">>> [--dry-run kích hoạt] Chỉ in kế hoạch, không gọi model. Dừng tại đây.")
        return

    if not needed_jds and not needed_cvs:
        print(">>> Toàn bộ dữ liệu đã được bóc tách hợp lệ theo schema hiện tại! Không có mục cần xử lý.")
        return

    backup_cache_file(args.jd_cache)
    backup_cache_file(args.cv_cache)

    if args.backend == "nvidia":
        backend: BaseExtractorBackend = NvidiaAPIBackend(
            model_name=args.model,
            rpm=args.rpm,
        )
    else:
        backend = VLLMBackend(model_name=args.model)

    start_time = time.time()
    jd_succ = 0
    jd_fail = 0
    cv_succ = 0
    cv_fail = 0

    def save_checkpoints():
        save_json(args.jd_cache, jd_cache)
        save_json(args.cv_cache, cv_cache)
        save_json(args.failed_file, failed_extractions)

    try:
        # Phase 1: JD Extraction
        if needed_jds:
            print(f"\n[PHASE 1] Xử lý {len(needed_jds)} Job Descriptions...")
            if args.backend == "nvidia":
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(process_jd_item, j_id, text, backend, args.model): j_id
                        for j_id, text in needed_jds
                    }
                    for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Trích xuất JD"), 1):
                        j_id, res, err = future.result()
                        if res is not None:
                            jd_cache[j_id] = res
                            jd_succ += 1
                            if j_id in failed_jds:
                                del failed_jds[j_id]
                        else:
                            jd_fail += 1
                            failed_jds[j_id] = {
                                "error": err,
                                "timestamp": time.time(),
                            }
                        if idx % 5 == 0 or idx == len(futures):
                            save_checkpoints()
            else:
                for idx, (j_id, text) in enumerate(tqdm(needed_jds, desc="Trích xuất JD (vLLM)"), 1):
                    _, res, err = process_jd_item(j_id, text, backend, args.model)
                    if res is not None:
                        jd_cache[j_id] = res
                        jd_succ += 1
                        if j_id in failed_jds:
                            del failed_jds[j_id]
                    else:
                        jd_fail += 1
                        failed_jds[j_id] = {"error": err, "timestamp": time.time()}
                    if idx % 5 == 0 or idx == len(needed_jds):
                        save_checkpoints()

        # Phase 2: CV Extraction
        if needed_cvs:
            print(f"\n[PHASE 2] Xử lý {len(needed_cvs)} Resumes / CVs...")
            if args.backend == "nvidia":
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(process_cv_item, c_hash, text, backend, args.model): c_hash
                        for c_hash, text in needed_cvs
                    }
                    for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Trích xuất CV"), 1):
                        c_hash, res, err = future.result()
                        if res is not None:
                            cv_cache[c_hash] = res
                            cv_succ += 1
                            if c_hash in failed_cvs:
                                del failed_cvs[c_hash]
                        else:
                            cv_fail += 1
                            failed_cvs[c_hash] = {
                                "error": err,
                                "timestamp": time.time(),
                            }
                        if idx % 5 == 0 or idx == len(futures):
                            save_checkpoints()
            else:
                for idx, (c_hash, text) in enumerate(tqdm(needed_cvs, desc="Trích xuất CV (vLLM)"), 1):
                    _, res, err = process_cv_item(c_hash, text, backend, args.model)
                    if res is not None:
                        cv_cache[c_hash] = res
                        cv_succ += 1
                        if c_hash in failed_cvs:
                            del failed_cvs[c_hash]
                    else:
                        cv_fail += 1
                        failed_cvs[c_hash] = {"error": err, "timestamp": time.time()}
                    if idx % 5 == 0 or idx == len(needed_cvs):
                        save_checkpoints()

    except KeyboardInterrupt:
        print("\n[CẢNH BÁO] Nhận tín hiệu ngắt (Ctrl+C). Đang lưu cache checkpoint...")
        save_checkpoints()
        print(">>> Đã lưu cache an toàn. Chạy lại script để tiếp tục.")
        return

    save_checkpoints()
    total_time = time.time() - start_time
    total_processed = jd_succ + jd_fail + cv_succ + cv_fail
    throughput = total_processed / max(total_time, 0.001)

    print("\n" + "=" * 75)
    print("BÁO CÁO KẾT QUẢ TRÍCH XUẤT:")

    summary_headers = ["Chỉ số", "JD", "CV", "Tổng cộng"]
    summary_rows = [
        ["Tổng số mẫu đầu vào", str(len(unique_jds)), str(len(unique_cvs)), str(len(unique_jds) + len(unique_cvs))],
        ["Mục tiêu trích xuất đợt này", str(len(needed_jds)), str(len(needed_cvs)), str(len(needed_jds) + len(needed_cvs))],
        ["Trích xuất thành công", str(jd_succ), str(cv_succ), str(jd_succ + cv_succ)],
        ["Thất bại", str(jd_fail), str(cv_fail), str(jd_fail + cv_fail)],
        ["Tổng số hợp lệ trong cache hiện tại", str(len(jd_cache)), str(len(cv_cache)), str(len(jd_cache) + len(cv_cache))],
        ["Thời gian thực thi", "-", "-", f"{total_time:.2f}s"],
        ["Tốc độ (Throughput)", "-", "-", f"{throughput:.2f} items/s"],
    ]
    print(to_markdown_table(summary_headers, summary_rows))

    # Major Group Distribution
    jd_majors = Counter(v.get("major_group", "unknown") for v in jd_cache.values() if isinstance(v, dict))
    cv_majors = Counter(v.get("major_group", "unknown") for v in cv_cache.values() if isinstance(v, dict))
    major_headers = ["Major Group", "JD Count", "CV Count"]
    major_rows = [
        [mg, str(jd_majors.get(mg, 0)), str(cv_majors.get(mg, 0))]
        for mg in MAJOR_GROUPS
        if jd_majors.get(mg, 0) > 0 or cv_majors.get(mg, 0) > 0
    ]
    if major_rows:
        print("\nPhân bố Major Group:")
        print(to_markdown_table(major_headers, major_rows))

    # Degree Distribution
    jd_degrees = Counter(v.get("min_degree", "none") for v in jd_cache.values() if isinstance(v, dict))
    cv_degrees = Counter(v.get("highest_degree", "none") for v in cv_cache.values() if isinstance(v, dict))
    degree_headers = ["Degree Level", "JD min_degree", "CV highest_degree"]
    degree_rows = [
        [deg, str(jd_degrees.get(deg, 0)), str(cv_degrees.get(deg, 0))]
        for deg in DEGREE_LEVELS
    ]
    print("\nPhân bố Bậc học (Degree):")
    print(to_markdown_table(degree_headers, degree_rows))
    print("=" * 75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hệ thống trích xuất thực thể tuyển dụng hợp nhất (API & vLLM).")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR, help="Thư mục chứa parquet splits.")
    parser.add_argument("--jd-cache", type=Path, default=JD_CACHE_PATH, help="Đường dẫn cache JD.")
    parser.add_argument("--cv-cache", type=Path, default=CV_CACHE_PATH, help="Đường dẫn cache CV.")
    parser.add_argument("--failed-file", type=Path, default=FAILED_EXTRACTIONS_PATH, help="Đường dẫn file lưu lỗi.")
    parser.add_argument("--backend", choices=["nvidia", "vllm"], default="nvidia", help="Backend suy luận.")
    parser.add_argument("--model", type=str, default=None, help="Tên model (mặc định lấy theo backend).")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng gọi song song cho API.")
    parser.add_argument("--rpm", type=float, default=35.0, help="Giới hạn số request/phút cho API.")
    parser.add_argument("--sample", type=int, default=None, help="Chỉ xử lý N JD và N CV đầu tiên cần trích xuất.")
    parser.add_argument("--only", choices=["jd", "cv", "all"], default="all", help="Chỉ chạy riêng JD hoặc CV.")
    parser.add_argument("--retry-failed", action="store_true", help="Chỉ chạy lại các mẫu trong failed_extractions.json.")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch xử lý, không gọi model.")
    parser.add_argument("--force", action="store_true", help="Trích xuất lại toàn bộ không quan tâm cache.")

    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_NVIDIA_MODEL if args.backend == "nvidia" else DEFAULT_VLLM_MODEL
    return args


if __name__ == "__main__":
    run_pipeline(parse_args())

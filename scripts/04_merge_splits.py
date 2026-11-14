import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import (
    CV_CACHE_PATH,
    JD_CACHE_PATH,
    PROCESSED_DIR,
)
from src.data.loader import get_split_path
from src.utils import compute_text_hash, load_json, to_markdown_table


def is_valid_jd_entities(ent: Optional[Dict[str, Any]]) -> bool:
    if not ent or not isinstance(ent, dict):
        return False
    required_keys = ["must_have_skills", "min_years_experience", "min_degree"]
    return all(k in ent for k in required_keys)


def is_valid_cv_entities(ent: Optional[Dict[str, Any]]) -> bool:
    if not ent or not isinstance(ent, dict):
        return False
    required_keys = ["skills", "total_years_experience", "highest_degree"]
    return all(k in ent for k in required_keys)


def process_split(
    split_name: str,
    data_dir: Path,
    jd_cache: Dict[str, Any],
    cv_cache: Dict[str, Any]
) -> Dict[str, Any]:
    parquet_path = data_dir / f"{split_name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    total_raw = len(df)
    output_path = get_split_path(split_name, data_dir)

    kept_records: List[Dict[str, Any]] = []
    dropped_count = 0
    missing_jd_count = 0
    missing_cv_count = 0

    for _, row in tqdm(df.iterrows(), total=total_raw, desc=f"Ghép tập {split_name}"):
        job_id = str(row["job_id"])
        resume_text = row["resume_text"]
        c_hash = compute_text_hash(resume_text)

        jd_ent = jd_cache.get(job_id)
        cv_ent = cv_cache.get(c_hash)

        has_valid_jd = is_valid_jd_entities(jd_ent)
        has_valid_cv = is_valid_cv_entities(cv_ent)

        if not has_valid_jd or not has_valid_cv:
            dropped_count += 1
            if not has_valid_jd:
                missing_jd_count += 1
            if not has_valid_cv:
                missing_cv_count += 1
            continue

        pair_id = f"{split_name}_{len(kept_records):05d}"
        label_text = str(row["label_text"]) if "label_text" in row else str(row["label"])
        record = {
            "pair_id": pair_id,
            "job_id": job_id,
            "label": int(row["label"]),
            "label_text": label_text,
            "job_description_text": row["job_description_text"],
            "resume_text": resume_text,
            "jd_entities": jd_ent,
            "cv_entities": cv_ent,
        }
        kept_records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    unique_jds = len({r["job_id"] for r in kept_records})
    label_dist = {0: 0, 1: 0, 2: 0}
    for r in kept_records:
        lbl = r["label"]
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    return {
        "split": split_name,
        "total_raw": total_raw,
        "kept": len(kept_records),
        "dropped": dropped_count,
        "missing_jd": missing_jd_count,
        "missing_cv": missing_cv_count,
        "unique_jds": unique_jds,
        "labels": f"0:{label_dist.get(0,0)} | 1:{label_dist.get(1,0)} | 2:{label_dist.get(2,0)}",
        "output_path": output_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Ghép cache thực thể JD/CV vào các split và lọc mẫu thiếu.")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR, help="Thư mục chứa dữ liệu processed.")
    parser.add_argument("--jd-cache", type=Path, default=JD_CACHE_PATH, help="Đường dẫn file cache JD.")
    parser.add_argument("--cv-cache", type=Path, default=CV_CACHE_PATH, help="Đường dẫn file cache CV.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Danh sách splits cần xử lý.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("BẮT ĐẦU QUY TRÌNH GHÉP CACHE VÀ XUẤT DATASET HOÀN CHỈNH (JSONL)")
    print(f"Thư mục dữ liệu: {args.data_dir}")
    print(f"Cache JD       : {args.jd_cache}")
    print(f"Cache CV       : {args.cv_cache}")
    print("=" * 75)

    jd_cache = load_json(args.jd_cache, {})
    cv_cache = load_json(args.cv_cache, {})
    print(f">>> Đã nạp Cache: {len(jd_cache):,} JD | {len(cv_cache):,} CV")

    start_time = time.time()
    results = []

    for split in args.splits:
        stats = process_split(split, args.data_dir, jd_cache, cv_cache)
        results.append(stats)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"TỔNG KẾT TIẾN TRÌNH GHÉP DỮ LIỆU (Thời gian: {elapsed:.2f}s):")

    headers = ["Tập Split", "Gốc (Raw)", "Hợp lệ (Kept)", "Bị loại (Drop)", "JD thiếu", "CV thiếu", "JD duy nhất", "Phân bố nhãn"]
    rows = [
        [
            r["split"],
            f"{r['total_raw']:,}",
            f"{r['kept']:,}",
            f"{r['dropped']:,}",
            f"{r['missing_jd']:,}",
            f"{r['missing_cv']:,}",
            f"{r['unique_jds']:,}",
            r["labels"]
        ]
        for r in results
    ]
    print(to_markdown_table(headers, rows))
    for r in results:
        print(f"-> Đã lưu: {r['output_path']}")
    print("=" * 75)


if __name__ == "__main__":
    main()

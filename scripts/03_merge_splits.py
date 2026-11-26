"""Ghép cache thực thể JD/CV vào raw_dataset.parquet để xuất data/processed/all.jsonl.

CẢNH BÁO: Chạy lại script này cần cache jd_cache.json và cv_cache.json đúng bản server,
nếu không all.jsonl sẽ bị ghi đè bằng entity cũ.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.config import (
    ALL_PAIRS_PATH,
    CV_CACHE_PATH,
    JD_CACHE_PATH,
    RAW_DIR,
)
from src.data.schemas import CVEntities, JDEntities, is_current
from src.utils import compute_text_hash, load_json, to_markdown_table


def merge_all(
    raw_path: Path = RAW_DIR / "raw_dataset.parquet",
    output_path: Path = ALL_PAIRS_PATH,
    jd_cache: Any = JD_CACHE_PATH,
    cv_cache: Any = CV_CACHE_PATH,
    strict: bool = True,
) -> Dict[str, Any]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file raw: {raw_path}")

    df = pd.read_parquet(raw_path)
    total_raw = len(df)

    jd_dict = jd_cache if isinstance(jd_cache, dict) else load_json(jd_cache, {})
    cv_dict = cv_cache if isinstance(cv_cache, dict) else load_json(cv_cache, {})
    print(f">>> Đã nạp Cache: {len(jd_dict):,} JD | {len(cv_dict):,} CV")

    kept_records: List[Dict[str, Any]] = []
    dropped_count = 0
    missing_jd_count = 0
    missing_cv_count = 0

    for idx, row in tqdm(df.iterrows(), total=total_raw, desc="Ghép toàn bộ all.jsonl"):
        job_id = str(row["job_id"])
        resume_text = row["resume_text"]
        c_hash = compute_text_hash(resume_text)

        jd_ent = jd_dict.get(job_id)
        cv_ent = cv_dict.get(c_hash)

        has_valid_jd = is_current(JDEntities, jd_ent)
        has_valid_cv = is_current(CVEntities, cv_ent)

        if not has_valid_jd or not has_valid_cv:
            dropped_count += 1
            if not has_valid_jd:
                missing_jd_count += 1
            if not has_valid_cv:
                missing_cv_count += 1
            continue

        pair_id = f"P{idx:05d}"
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

    if dropped_count > 0:
        msg = (f"Tổng cộng {dropped_count}/{total_raw} mẫu thiếu entity hợp lệ "
               f"({missing_jd_count} thiếu JD, {missing_cv_count} thiếu CV). Chạy lại scripts/02_extract_entities.py.")
        if strict:
            sys.exit(f"[LỖI] {msg} Dùng --no-strict nếu chấp nhận bỏ các mẫu này.")
        print(f"[CẢNH BÁO] {msg}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    unique_jds = len({r["job_id"] for r in kept_records})
    label_dist = {0: 0, 1: 0, 2: 0}
    for r in kept_records:
        lbl = r["label"]
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    return {
        "split": "all",
        "total_raw": total_raw,
        "kept": len(kept_records),
        "dropped": dropped_count,
        "missing_jd": missing_jd_count,
        "missing_cv": missing_cv_count,
        "unique_jds": unique_jds,
        "labels": f"0:{label_dist.get(0,0)} | 1:{label_dist.get(1,0)} | 2:{label_dist.get(2,0)}",
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Ghép cache thực thể JD/CV vào raw_dataset.parquet.")
    parser.add_argument("--raw-path", type=Path, default=RAW_DIR / "raw_dataset.parquet")
    parser.add_argument("--output-path", type=Path, default=ALL_PAIRS_PATH)
    parser.add_argument("--jd-cache", type=Path, default=JD_CACHE_PATH)
    parser.add_argument("--cv-cache", type=Path, default=CV_CACHE_PATH)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    start_time = time.time()
    stats = merge_all(args.raw_path, args.output_path, args.jd_cache, args.cv_cache, strict=args.strict)
    elapsed = time.time() - start_time

    headers = ["Tập dữ liệu", "Gốc (Raw)", "Hợp lệ (Kept)", "Bị loại (Drop)", "JD thiếu", "CV thiếu", "JD duy nhất", "Phân bố nhãn"]
    rows = [[
        stats["split"],
        f"{stats['total_raw']:,}",
        f"{stats['kept']:,}",
        f"{stats['dropped']:,}",
        f"{stats['missing_jd']:,}",
        f"{stats['missing_cv']:,}",
        f"{stats['unique_jds']:,}",
        stats["labels"]
    ]]
    print("\n" + to_markdown_table(headers, rows))
    print(f"-> Đã lưu: {stats['output_path']} ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()

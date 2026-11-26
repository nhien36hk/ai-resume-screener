"""Số liệu mô tả dữ liệu cho tiểu luận (Bảng 3.2, Bảng 4.3, tỉ lệ CV lặp, 2 cặp minh hoạ).

Không ảnh hưởng kết quả thực nghiệm. Ghi artifacts/results/data_stats.md (+ .json).
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config import (CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL, DEFAULT_SEEDS, FAILED_EXTRACTIONS_PATH,
                        ID_TO_LABEL, MAX_SEQ_LENGTH, RAW_DIR, RESULTS_DIR)
from src.data.loader import SCHEMES, load_all, select_split, split_path
from src.utils import compute_text_hash, load_json, save_json, to_markdown_table


def n_chunks(n_tokens: int) -> int:
    """Số đoạn của chunk_text (200 token, chồng 50) suy từ số token, khỏi decode lại."""
    if n_tokens <= CHUNK_SIZE:
        return 1
    stride = CHUNK_SIZE - CHUNK_OVERLAP
    return int(np.ceil((n_tokens - CHUNK_SIZE) / stride)) + 1


def describe(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.0f} / {np.median(v):.0f} / {np.percentile(v, 95):.0f}"


def dataset_table(df: pd.DataFrame, tokenizer) -> tuple:
    cvs = df.drop_duplicates("resume_text")
    jds = df.drop_duplicates("job_id")
    cv_tokens = np.array([len(tokenizer.encode(t, add_special_tokens=False)) for t in cvs["resume_text"]])
    jd_tokens = np.array([len(tokenizer.encode(t, add_special_tokens=False)) for t in jds["job_description_text"]])
    raw_path = RAW_DIR / "raw_dataset.parquet"
    raw = pd.read_parquet(raw_path) if raw_path.exists() else None
    failed = load_json(FAILED_EXTRACTIONS_PATH, {}) or {}
    must_after = [len(e.get("must_have_skills", [])) for e in jds["jd_entities"]]
    must_before = [len(e.get("must_have_skills_orig", e.get("must_have_skills", []))) for e in jds["jd_entities"]]
    labels = df["label"].value_counts().sort_index()
    stats = {
        "n_pairs": len(df), "n_jd": int(df["job_id"].nunique()), "n_cv_texts": int(df["resume_text"].nunique()),
        "raw_pairs": None if raw is None else len(raw),
        "raw_jd": None if raw is None else int(raw["job_description_text"].nunique()),
        "raw_cv_texts": None if raw is None else int(raw["resume_text"].nunique()),
        "failed_jd": len(failed.get("jds", [])), "failed_cv": len(failed.get("cvs", [])),
        "labels": {ID_TO_LABEL.get(int(k), str(k)): int(v) for k, v in labels.items()},
        "cv_words": describe(cvs["resume_text"].str.split().str.len()),
        "jd_words": describe(jds["job_description_text"].str.split().str.len()),
        "cv_tokens": describe(cv_tokens), "jd_tokens": describe(jd_tokens),
        "pct_cv_over_max_seq": float((cv_tokens > MAX_SEQ_LENGTH).mean() * 100),
        "pct_jd_over_max_seq": float((jd_tokens > MAX_SEQ_LENGTH).mean() * 100),
        "cv_chunks_mean": float(np.mean([n_chunks(n) for n in cv_tokens])),
        "jd_chunks_mean": float(np.mean([n_chunks(n) for n in jd_tokens])),
        "must_before_02b_mean": float(np.mean(must_before)), "must_after_02b_mean": float(np.mean(must_after)),
        "pct_cv_zero_years": float((cvs["cv_entities"].map(lambda e: float(e.get("total_years_experience") or 0) == 0)).mean() * 100),
    }
    rows = [
        ["Số cặp (CV, JD)", f"{stats['n_pairs']:,}" + (f" (gốc {stats['raw_pairs']:,})" if raw is not None else "")],
        ["Số JD", f"{stats['n_jd']}" + (f" (gốc {stats['raw_jd']})" if raw is not None else "")],
        ["Số văn bản CV khác nhau", f"{stats['n_cv_texts']}" + (f" (gốc {stats['raw_cv_texts']})" if raw is not None else "")],
        ["JD / CV trích xuất lỗi bị loại", f"{stats['failed_jd']} / {stats['failed_cv']}"],
        ["Phân bố nhãn", ", ".join(f"{k}: {v:,} ({v / len(df) * 100:.1f}%)" for k, v in stats["labels"].items())],
        ["Số từ CV (TB / trung vị / p95)", stats["cv_words"]],
        ["Số từ JD (TB / trung vị / p95)", stats["jd_words"]],
        ["Số token CV, tokenizer MiniLM (TB / trung vị / p95)", stats["cv_tokens"]],
        ["Số token JD (TB / trung vị / p95)", stats["jd_tokens"]],
        [f"% CV dài hơn {MAX_SEQ_LENGTH} token", f"{stats['pct_cv_over_max_seq']:.1f}%"],
        [f"% JD dài hơn {MAX_SEQ_LENGTH} token", f"{stats['pct_jd_over_max_seq']:.1f}%"],
        [f"Số đoạn TB mỗi CV / JD ({CHUNK_SIZE} token, chồng {CHUNK_OVERLAP})", f"{stats['cv_chunks_mean']:.2f} / {stats['jd_chunks_mean']:.2f}"],
        ["Số kỹ năng bắt buộc TB mỗi JD (trước / sau bước 02b)", f"{stats['must_before_02b_mean']:.2f} / {stats['must_after_02b_mean']:.2f}"],
        ["% CV không trích được số năm kinh nghiệm (= 0)", f"{stats['pct_cv_zero_years']:.1f}%"],
    ]
    return stats, to_markdown_table(["Đặc điểm", "Giá trị"], rows)


def split_table(df: pd.DataFrame, seeds) -> tuple:
    rows, records = [], []
    hashes = df["resume_text"].map(lambda t: compute_text_hash(t)[:12])
    for scheme in SCHEMES:
        for seed in seeds:
            if not split_path(seed, scheme).exists():
                continue
            parts = {name: select_split(df.assign(cv_id=hashes), name, seed, scheme) for name in ("train", "val", "test")}
            train_cvs = set(parts["train"]["cv_id"])
            cv_seen = float(parts["test"]["cv_id"].isin(train_cvs).mean() * 100)
            for name, part in parts.items():
                dist = part["label"].value_counts().reindex([0, 1, 2], fill_value=0)
                rec = {"scheme": scheme, "seed": seed, "split": name, "n_pairs": len(part), "n_jd": int(part["job_id"].nunique()),
                       "labels": dist.tolist(), "pct_test_cv_in_train": cv_seen if name == "test" else None}
                records.append(rec)
                rows.append([scheme, str(seed), name, str(len(part)), str(rec["n_jd"]),
                             " / ".join(f"{v} ({v / max(len(part), 1) * 100:.0f}%)" for v in dist),
                             f"{cv_seen:.1f}%" if name == "test" else ""])
    md = to_markdown_table(["Cách chia", "Seed", "Tập", "Số cặp", "Số JD", "Nhãn 0 / 1 / 2", "% cặp Test có CV đã ở Train"], rows)
    return records, md


def example_pairs(df: pd.DataFrame, n_chars: int = 400) -> str:
    out = []
    for label in (2, 0):
        r = df[df["label"] == label].iloc[0]
        jd, cv = r["jd_entities"], r["cv_entities"]
        out.append(
            f"### Cặp minh hoạ nhãn {label} ({ID_TO_LABEL.get(label, label)}), pair_id {r['pair_id']}\n\n"
            f"- JD: {jd.get('job_title', '')}; must: {', '.join(jd.get('must_have_skills', []))}; "
            f"năm KN tối thiểu: {jd.get('min_years_experience')}; bậc học: {jd.get('min_degree')}\n"
            f"- CV: {cv.get('candidate_title', '')}; {len(cv.get('skills', []))} kỹ năng; "
            f"{cv.get('total_years_experience')} năm KN; bậc học: {cv.get('highest_degree')}\n\n"
            f"> JD: {r['job_description_text'][:n_chars].strip()}…\n>\n> CV: {r['resume_text'][:n_chars].strip()}…\n"
        )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    df = load_all()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EMBEDDING_MODEL)
    stats, md_data = dataset_table(df, tokenizer)
    splits, md_split = split_table(df, args.seeds)
    md = ("# Số liệu dữ liệu\n\n## Bảng 3.2: Đặc điểm bộ dữ liệu\n\n" + md_data
          + "\n\n## Bảng 4.3: Kích thước các tập theo seed\n\n" + md_split
          + "\n\n## Cặp minh hoạ (cắt ngắn)\n\n" + example_pairs(df) + "\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_stats.md").write_text(md, encoding="utf-8")
    save_json(RESULTS_DIR / "data_stats.json", {"dataset": stats, "splits": splits})
    print(md)


if __name__ == "__main__":
    main()

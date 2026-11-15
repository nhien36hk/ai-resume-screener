"""Gộp 3 file train/val/test (jsonl và parquet) thành all duy nhất và kiểm tra 1-1 từng dòng."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import ALL_PAIRS_PATH, FEATURES_DIR, PROCESSED_DIR


def merge_and_verify() -> None:
    splits = ["train", "val", "test"]

    # 1. Kiểm tra hoặc gộp jsonl
    if ALL_PAIRS_PATH.exists():
        print(f">>> 1. File {ALL_PAIRS_PATH.name} đã tồn tại sẵn.")
    else:
        jsonl_files = [PROCESSED_DIR / f"{s}.jsonl" for s in splits]
        if all(p.exists() for p in jsonl_files):
            print(">>> 1. Đang gộp train.jsonl + val.jsonl + test.jsonl -> all.jsonl...")
            lines = []
            for p in jsonl_files:
                with open(p, "r", encoding="utf-8") as f:
                    lines.extend(f.readlines())
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            with open(ALL_PAIRS_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"    -> Đã ghi {len(lines):,} dòng vào {ALL_PAIRS_PATH.name}")
        else:
            raise FileNotFoundError(f"Không tìm thấy {ALL_PAIRS_PATH} hoặc bộ 3 file jsonl trong {PROCESSED_DIR}")

    # 2. Kiểm tra hoặc gộp parquet
    feat_dir = FEATURES_DIR / "minilm_pretrained"
    feat_files = [feat_dir / f"{s}.parquet" for s in splits]
    all_parquet_path = feat_dir / "all.parquet"

    if all_parquet_path.exists():
        print(f">>> 2. File {all_parquet_path.name} đã tồn tại sẵn.")
    elif all(p.exists() for p in feat_files):
        print(">>> 2. Đang gộp train.parquet + val.parquet + test.parquet -> all.parquet...")
        dfs = [pd.read_parquet(p) for p in feat_files]
        merged_df = pd.concat(dfs, ignore_index=True)
        feat_dir.mkdir(parents=True, exist_ok=True)
        merged_df.to_parquet(all_parquet_path, index=False)
        print(f"    -> Đã ghi {len(merged_df):,} dòng vào {all_parquet_path.name}")
    else:
        print(f">>> 2. Chưa có cache parquet trên server, đang tự động encode từ {ALL_PAIRS_PATH.name}...")
        from sentence_transformers import SentenceTransformer
        from src.config import DEFAULT_EMBEDDING_MODEL, MAX_SEQ_LENGTH
        from src.pipeline import precompute_all_features
        
        df_all = pd.read_json(ALL_PAIRS_PATH, lines=True)
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        model.max_seq_length = MAX_SEQ_LENGTH
        feat_df = precompute_all_features(df_all, model)
        feat_dir.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(all_parquet_path, index=False)
        print(f"    -> Đã tính toán và lưu {len(feat_df):,} dòng vào {all_parquet_path.name}")

    # 3. Kiểm tra 1-1 từng dòng giữa all.jsonl và all.parquet
    print(">>> 3. Đang đối soát 1-1 từng dòng giữa all.jsonl và all.parquet...")
    df_jl = pd.read_json(ALL_PAIRS_PATH, lines=True)
    df_pq = pd.read_parquet(all_parquet_path)

    assert len(df_jl) == len(df_pq), f"LỆCH SỐ LƯỢNG DÒNG: jsonl={len(df_jl)} != parquet={len(df_pq)}"
    n = len(df_jl)

    assert (df_jl["pair_id"].values == df_pq["pair_id"].values).all(), "LỆCH pair_id từng dòng!"
    assert (df_jl["job_id"].astype(str).values == df_pq["job_id"].astype(str).values).all(), "LỆCH job_id từng dòng!"
    assert (df_jl["label"].astype(int).values == df_pq["label"].astype(int).values).all(), "LỆCH label từng dòng!"

    unique_jds = df_jl["job_id"].nunique()
    print("\n" + "=" * 60)
    print("XÁC NHẬN TOÀN VẸN DỮ LIỆU THÀNH CÔNG (100% MATCH)")
    print("=" * 60)
    print(f"- Tổng số cặp CV-JD   : {n:,}")
    print(f"- Tổng số JD duy nhất : {unique_jds:,}")
    print(f"- Khớp chính xác 100% : pair_id, job_id, label theo từng dòng.")
    print("=" * 60)
    all_parquet_path.touch()


if __name__ == "__main__":
    merge_and_verify()


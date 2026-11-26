import importlib
from pathlib import Path
import json

import pandas as pd
import pytest

from src.config import FEATURES_DIR, SPLITS_DIR
from src.data.loader import load_all, load_split, split_ids

make_splits = importlib.import_module("scripts.04_make_splits").make_splits


def test_splits_integrity_across_seeds():
    """Với 3 seed bất kỳ: không trùng JD, hợp lại đủ mọi JD của all.jsonl, tỷ lệ xấp xỉ 70/15/15."""
    df_all = load_all()
    all_jds = set(df_all["job_id"].unique())
    total_jds = len(all_jds)

    for seed in [0, 2, 4]:
        splits = split_ids(seed)
        train_jds = set(splits["train"])
        val_jds = set(splits["val"])
        test_jds = set(splits["test"])

        assert len(train_jds & val_jds) == 0, f"Trùng Train-Val ở seed {seed}"
        assert len(train_jds & test_jds) == 0, f"Trùng Train-Test ở seed {seed}"
        assert len(val_jds & test_jds) == 0, f"Trùng Val-Test ở seed {seed}"
        assert (train_jds | val_jds | test_jds) == all_jds

        assert abs(len(train_jds) / total_jds - 0.70) < 0.05
        assert abs(len(val_jds) / total_jds - 0.15) < 0.05
        assert abs(len(test_jds) / total_jds - 0.15) < 0.05


def test_idempotent_splits_no_overwrite():
    """Cùng một seed tạo ra cùng file index; chạy lại không ghi đè."""
    seed0_path = SPLITS_DIR / "seed0.json"
    mtime_before = seed0_path.stat().st_mtime
    make_splits(seeds=[0])
    mtime_after = seed0_path.stat().st_mtime
    assert mtime_before == mtime_after


def test_split_and_feature_pair_ids_match():
    """load_split(name, seed) và feature tables cắt theo seed trả về cùng tập pair_id."""
    seed = 0
    for name in ["train", "val", "test"]:
        df_split = load_split(name, seed=seed)
        split_pair_ids = set(df_split["pair_id"])

        all_feat_path = FEATURES_DIR / "minilm_pretrained" / "all.parquet"
        df_feat = pd.read_parquet(all_feat_path)
        jds = set(split_ids(seed)[name])
        feat_pair_ids = set(df_feat[df_feat["job_id"].isin(jds)]["pair_id"])

        assert split_pair_ids == feat_pair_ids, f"Lệch pair_id ở split {name} seed {seed}"


def test_all_jsonl_and_all_parquet_exact_alignment():
    """Kiểm tra all.jsonl và all.parquet khớp 1-1 tuyệt đối: cùng thứ tự dòng, pair_id, job_id, label."""
    df_jl = load_all()
    df_pq = pd.read_parquet(FEATURES_DIR / "minilm_pretrained" / "all.parquet")

    assert len(df_jl) == len(df_pq) == 8000
    assert (df_jl["pair_id"].values == df_pq["pair_id"].values).all(), "Lệch pair_id giữa jsonl và parquet"
    assert (df_jl["job_id"].astype(str).values == df_pq["job_id"].astype(str).values).all(), "Lệch job_id giữa jsonl và parquet"
    assert (df_jl["label"].astype(int).values == df_pq["label"].astype(int).values).all(), "Lệch label giữa jsonl và parquet"



def test_strict_assignment_keeps_jd_and_cv_in_one_split():
    """Chia chặt: không JD nào và không CV nào nằm ở 2 tập; cùng seed -> cùng kết quả."""
    strict_assignment = importlib.import_module("scripts.04_make_splits").strict_assignment
    from src.utils import compute_text_hash
    df = load_all()
    assigned = strict_assignment(df, seed=0)
    kept = df.assign(split=assigned, cv_id=df["resume_text"].map(lambda t: compute_text_hash(t)[:12])).dropna(subset=["split"])
    assert kept.groupby("job_id")["split"].nunique().max() == 1
    assert kept.groupby("cv_id")["split"].nunique().max() == 1
    assert set(kept["split"]) == {"train", "val", "test"}
    assert 0.3 < len(kept) / len(df) < 0.8
    assert assigned.equals(strict_assignment(df, seed=0))

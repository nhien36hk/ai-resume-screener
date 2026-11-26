"""Tạo file index chia tập dữ liệu cho từng seed.

--scheme jd (mặc định): GroupShuffleSplit theo job_id, lưu danh sách job_id.
--scheme strict: chia JD 70/15/15 và CV 70/15/15 độc lập (cùng seed), chỉ giữ cặp có JD và CV cùng tập
  -> mỗi JD và mỗi CV chỉ nằm ở một tập (theo góp ý GVHD mục 3.4.4). Lưu danh sách pair_id.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_PAIRS_PATH, DEFAULT_SEEDS
from src.data.loader import split_path
from src.data.splitter import split_by_job_group, verify_splits
from src.utils import compute_text_hash, save_json

SPLIT_NAMES = ("train", "val", "test")


def group_assignment(df: pd.DataFrame, group_col: str, seed: int) -> dict:
    """{giá trị nhóm: tên tập} — dùng lại split_by_job_group (hàm này nhóm theo cột job_id)."""
    parts = split_by_job_group(df[[group_col]].rename(columns={group_col: "job_id"}), random_state=seed)
    return {g: name for name, part in zip(SPLIT_NAMES, parts) for g in part["job_id"].unique()}


def strict_assignment(df: pd.DataFrame, seed: int) -> pd.Series:
    """Tên tập của mỗi cặp, NaN nếu JD và CV rơi vào hai tập khác nhau (cặp bị bỏ)."""
    cv_id = df["resume_text"].map(lambda t: compute_text_hash(t)[:12])
    jd_set = df["job_id"].map(group_assignment(df, "job_id", seed))
    cv_set = cv_id.map(group_assignment(df.assign(cv_id=cv_id), "cv_id", seed))
    return jd_set.where(jd_set == cv_set)


def make_splits(seeds=DEFAULT_SEEDS, scheme: str = "jd", force: bool = False):
    df = pd.read_json(ALL_PAIRS_PATH, lines=True)
    for seed in seeds:
        out_path = split_path(seed, scheme)
        if out_path.exists() and not force:
            print(f"Bỏ qua seed {seed}: {out_path} đã tồn tại.")
            continue

        if scheme == "jd":
            train_df, val_df, test_df = split_by_job_group(df[["job_id"]], random_state=seed)
            verify_splits(train_df, val_df, test_df)
            payload = {name: sorted(part["job_id"].unique().tolist()) for name, part in zip(SPLIT_NAMES, (train_df, val_df, test_df))}
        else:
            assigned = strict_assignment(df, seed)
            payload = {name: sorted(df.loc[assigned == name, "pair_id"].tolist()) for name in SPLIT_NAMES}
            payload["dropped"] = int(assigned.isna().sum())
            for name in SPLIT_NAMES:
                part = df[assigned == name]
                print(f"  seed {seed} {name}: {len(part)} cặp, {part['job_id'].nunique()} JD, "
                      f"nhãn {part['label'].value_counts().sort_index().to_dict()}")
            print(f"  seed {seed}: giữ {len(df) - payload['dropped']}/{len(df)} cặp, bỏ {payload['dropped']}")

        save_json(out_path, payload)
        print(f"Đã tạo {out_path}: " + ", ".join(f"{n}={len(payload[n])}" for n in SPLIT_NAMES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd")
    parser.add_argument("--force", action="store_true", help="Ghi đè file index đã có.")
    args = parser.parse_args()
    make_splits(args.seeds, args.scheme, args.force)

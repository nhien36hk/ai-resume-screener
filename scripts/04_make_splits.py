"""Tạo file index chia tập dữ liệu (GroupShuffleSplit theo job_id) cho từng seed."""
import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_PAIRS_PATH, DEFAULT_SEEDS, SPLITS_DIR
from src.data.splitter import split_by_job_group, verify_splits
from src.utils import save_json


def make_splits(seeds=DEFAULT_SEEDS):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(ALL_PAIRS_PATH, lines=True)[["job_id"]]

    for seed in seeds:
        out_path = SPLITS_DIR / f"seed{seed}.json"
        if out_path.exists():
            print(f"Bỏ qua seed {seed}: {out_path.name} đã tồn tại.")
            continue

        train_df, val_df, test_df = split_by_job_group(df, random_state=seed)
        verify_splits(train_df, val_df, test_df)

        payload = {
            "train": sorted(train_df["job_id"].unique().tolist()),
            "val": sorted(val_df["job_id"].unique().tolist()),
            "test": sorted(test_df["job_id"].unique().tolist()),
        }
        save_json(out_path, payload)
        print(f"Đã tạo {out_path.name}: Train={len(payload['train'])}, Val={len(payload['val'])}, Test={len(payload['test'])} JDs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    make_splits(args.seeds)


"""KTC 95% bootstrap theo cặp cho hiệu số giữa hai mô hình (tiểu luận 3.8.10), gộp các seed đã chạy.

Đọc RESULTS_DIR/<scheme>/seed*/preds/<mã>.parquet do E1–E5 ghi ra.
Macro-F1, QWK: lấy mẫu lại cặp; NDCG@10: lấy mẫu lại JD.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import RESULTS_DIR
from src.metrics import compute_macro_f1, compute_ndcg_at_10, compute_qwk, paired_bootstrap_ci
from src.utils import save_json, to_markdown_table

COMPARISONS = [("M3b", "M2"), ("M3b", "M1"), ("M3b", "M3a"), ("M4", "M3b"), ("M4", "M2"), ("M5", "M4"), ("M5", "M2")]


def _metric_f1(d):
    return compute_macro_f1(d["label"], d["pred"])


def _metric_qwk(d):
    return compute_qwk(d["label"], d["pred"])


def _metric_ndcg(d):
    return compute_ndcg_at_10(d["job_id"], d["label"], d["score"])


METRICS = {  # tên -> (hàm trên DataFrame preds, lấy mẫu theo JD?)
    "Macro-F1": (_metric_f1, False),
    "QWK": (_metric_qwk, False),
    "NDCG@10": (_metric_ndcg, True),
}


def load_pairs(scheme: str, code_a: str, code_b: str):
    """Mỗi seed có đủ dự đoán của cả hai mô hình -> một cặp (df_a, df_b) cùng thứ tự pair_id."""
    pairs, seeds = [], []
    for seed_dir in sorted((RESULTS_DIR / scheme).glob("seed*")):
        fa, fb = seed_dir / "preds" / f"{code_a}.parquet", seed_dir / "preds" / f"{code_b}.parquet"
        if not (fa.exists() and fb.exists()):
            continue
        a = pd.read_parquet(fa).sort_values("pair_id").reset_index(drop=True)
        b = pd.read_parquet(fb).sort_values("pair_id").reset_index(drop=True)
        assert a["pair_id"].equals(b["pair_id"]), f"{seed_dir.name}: {code_a} và {code_b} khác tập Test"
        pairs.append((a, b))
        seeds.append(seed_dir.name)
    return pairs, seeds


def _eval_task(task: tuple) -> dict:
    scheme, code_a, code_b, name, n_boot = task
    pairs, seeds = load_pairs(scheme, code_a, code_b)
    fn, by_jd = METRICS[name]
    point, lo, hi = paired_bootstrap_ci(pairs, fn, n_boot=n_boot, by_jd=by_jd)
    return {
        "a": code_a,
        "b": code_b,
        "metric": name,
        "diff": point,
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(lo > 0 or hi < 0),
        "seeds": seeds,
    }


def run_bootstrap(scheme: str = "jd", n_boot: int = 1000) -> list:
    tasks = []
    for code_a, code_b in COMPARISONS:
        pairs, _ = load_pairs(scheme, code_a, code_b)
        if not pairs:
            print(f"Bỏ qua {code_a} − {code_b}: thiếu file dự đoán")
            continue
        for name in METRICS:
            tasks.append((scheme, code_a, code_b, name, n_boot))

    out = []
    if tasks:
        n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        n_workers = max(1, min(n_cpus, len(tasks)))
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            out = list(pool.map(_eval_task, tasks))
        for r in out:
            print(f"{r['a']} − {r['b']} | {r['metric']:8s}: {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] ({len(r['seeds'])} seed)")

    out_dir = RESULTS_DIR / scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "bootstrap.json", {"scheme": scheme, "n_boot": n_boot, "results": out})
    rows = [[f"{r['a']} − {r['b']}", r["metric"], f"{r['diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
             "Có" if r["significant"] else "Không", str(len(r["seeds"]))] for r in out]
    md = to_markdown_table(["So sánh", "Chỉ số", "Hiệu số", "KTC 95%", "Khác biệt rõ (KTC không chứa 0)", "Số seed"], rows)
    with open(out_dir / "bootstrap.md", "w", encoding="utf-8") as f:
        f.write(f"# KTC 95% bootstrap theo cặp ({n_boot} lần, chia {scheme})\n\n" + md + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    run_bootstrap(args.scheme, args.n_boot)

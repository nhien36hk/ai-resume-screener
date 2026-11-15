import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import RESULTS_DIR
from src.data.loader import load_split
from src.metrics import evaluate_all
from src.scoring import scale_p5_p95
from src.tuning import search_best_thresholds
from src.utils import save_json, to_markdown_table


def run_e2(seed: int) -> dict:
    print(f"=== CHẠY THỰC NGHIỆM E2: BASELINE TF-IDF (Seed {seed}) ===")
    df_train = load_split("train", seed)
    df_val = load_split("val", seed)
    df_test = load_split("test", seed)

    unique_corpus = list(pd.unique(df_train["job_description_text"])) + list(pd.unique(df_train["resume_text"]))
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    vectorizer.fit(unique_corpus)

    def calc_sims(df):
        jd_mat = vectorizer.transform(df["job_description_text"].tolist())
        cv_mat = vectorizer.transform(df["resume_text"].tolist())
        return np.asarray(jd_mat.multiply(cv_mat).sum(axis=1)).flatten()

    train_sims = calc_sims(df_train)
    percentile_p5, percentile_p95 = float(np.percentile(train_sims, 5)), float(np.percentile(train_sims, 95))

    val_sims = calc_sims(df_val)
    val_scores = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in val_sims], dtype=np.float32)
    y_val = df_val["label"].to_numpy()

    best_t1, best_t2, val_f1 = search_best_thresholds(val_scores, y_val)
    print(f"Ngưỡng tối ưu E2 (seed {seed}): t1={best_t1:.1f}, t2={best_t2:.1f} (Macro-F1 Val: {val_f1:.4f})")

    test_sims = calc_sims(df_test)
    test_scores = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in test_sims], dtype=np.float32)
    y_test = df_test["label"].to_numpy()

    test_preds = np.zeros_like(test_scores, dtype=int)
    test_preds[test_scores >= best_t1] = 1
    test_preds[test_scores >= best_t2] = 2

    metrics = evaluate_all(
        y_true=y_test,
        y_pred=test_preds,
        continuous_scores=test_scores,
        job_ids=df_test["job_id"].to_numpy(),
    )

    headers = ["Mô hình", "Ngưỡng (t1, t2)", "QWK", "Macro-F1", "MAE", "Spearman", "NDCG@10"]
    row = [
        f"E2: Baseline TF-IDF (seed {seed})",
        f"({best_t1:.0f}, {best_t2:.0f})",
        f"{metrics['qwk']:.4f}",
        f"{metrics['macro_f1']:.4f}",
        f"{metrics['mae']:.4f}",
        f"{metrics.get('spearman', 0.0):.4f}",
        f"{metrics.get('ndcg_10', 0.0):.4f}",
    ]
    md_table = to_markdown_table(headers, [row])
    print("\n" + md_table + "\n")

    out_dir = RESULTS_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"seed": seed, "thresholds": (best_t1, best_t2), "scaler": (percentile_p5, percentile_p95), "metrics": metrics}
    save_json(out_dir / "e2.json", res)
    with open(out_dir / "e2.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E2: Baseline TF-IDF (Seed {seed})\n\n" + md_table + "\n")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed phân chia dữ liệu.")
    args = parser.parse_args()
    run_e2(args.seed)

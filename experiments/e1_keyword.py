import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import RESULTS_DIR
from src.data.loader import load_split
from src.features.skills import is_exact_match
from src.metrics import compute_all_metrics
from src.tuning import search_best_thresholds
from src.utils import save_json, to_markdown_table


def score_cv_keywords(must_skills, nice_skills, cv_text):
    if not must_skills and not nice_skills:
        return 0.0
    total_w = len(must_skills) * 2.0 + len(nice_skills) * 1.0
    matched_w = sum(2.0 for s in must_skills if is_exact_match(s, cv_text))
    matched_w += sum(1.0 for s in nice_skills if is_exact_match(s, cv_text))
    return 100.0 * (matched_w / total_w)


def run_e1(seed: int) -> dict:
    print(f"=== CHẠY THỰC NGHIỆM E1: BASELINE KEYWORD MATCH (Seed {seed}) ===")
    df_val = load_split("val", seed)
    df_test = load_split("test", seed)

    val_scores = np.array([
        score_cv_keywords(
            r["jd_entities"].get("must_have_skills", []),
            r["jd_entities"].get("nice_to_have_skills", []),
            r["resume_text"],
        )
        for _, r in df_val.iterrows()
    ], dtype=np.float32)

    y_val = df_val["label"].to_numpy()
    best_t1, best_t2, val_f1 = search_best_thresholds(val_scores, y_val)
    print(f"Ngưỡng tối ưu E1 (seed {seed}): t1={best_t1:.1f}, t2={best_t2:.1f} (Macro-F1 Val: {val_f1:.4f})")

    test_scores = np.array([
        score_cv_keywords(
            r["jd_entities"].get("must_have_skills", []),
            r["jd_entities"].get("nice_to_have_skills", []),
            r["resume_text"],
        )
        for _, r in df_test.iterrows()
    ], dtype=np.float32)

    y_test = df_test["label"].to_numpy()
    test_preds = np.zeros_like(test_scores, dtype=int)
    test_preds[test_scores >= best_t1] = 1
    test_preds[test_scores >= best_t2] = 2

    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=test_preds,
        continuous_scores=test_scores,
        job_ids=df_test["job_id"].to_numpy(),
    )

    headers = ["Mô hình", "Ngưỡng (t1, t2)", "QWK", "Macro-F1", "MAE", "Spearman", "NDCG@10"]
    row = [
        f"E1: Baseline Keyword (seed {seed})",
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
    res = {"seed": seed, "thresholds": (best_t1, best_t2), "metrics": metrics}
    save_json(out_dir / "e1.json", res)
    with open(out_dir / "e1.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E1: Baseline Keyword (Seed {seed})\n\n" + md_table + "\n")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed phân chia dữ liệu.")
    args = parser.parse_args()
    run_e1(args.seed)

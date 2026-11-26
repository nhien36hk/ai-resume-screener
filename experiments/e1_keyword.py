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
from src.metrics import compute_all_metrics, compute_spearman
from src.tuning import search_best_thresholds, to_labels
from src.utils import save_json, save_preds, to_markdown_table


def score_cv_keywords(must_skills, nice_skills, cv_text):
    """Tiểu luận 3.7.1: 100 × (số kỹ năng JD có nguyên văn trong CV) / (tổng số kỹ năng JD).
    Không trọng số, không bảng đồng nghĩa (synonyms={}), không độ tương đồng ngữ nghĩa."""
    skills = list(dict.fromkeys(must_skills + nice_skills))
    if not skills:
        return 0.0
    return 100.0 * sum(is_exact_match(s, cv_text, synonyms={}) for s in skills) / len(skills)


def run_e1(seed: int, scheme: str = "jd") -> dict:
    print(f"=== CHẠY THỰC NGHIỆM E1: BASELINE KEYWORD MATCH (Seed {seed}, chia {scheme}) ===")
    df_val = load_split("val", seed, scheme)
    df_test = load_split("test", seed, scheme)

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
    test_preds = to_labels(test_scores, best_t1, best_t2)

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

    out_dir = RESULTS_DIR / scheme / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_preds(out_dir, "M1", df_test, test_scores, test_preds)
    res = {"seed": seed, "scheme": scheme, "thresholds": (best_t1, best_t2), "val_macro_f1": val_f1, "val_spearman": compute_spearman(y_val, val_scores), "metrics": metrics}
    save_json(out_dir / "e1.json", res)
    with open(out_dir / "e1.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E1: Baseline Keyword (Seed {seed})\n\n" + md_table + "\n")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed phân chia dữ liệu.")
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd", help="Cách chia dữ liệu.")
    args = parser.parse_args()
    run_e1(args.seed, args.scheme)

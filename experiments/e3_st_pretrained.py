from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL, MAX_SEQ_LENGTH
from src.metrics import evaluate_all
from src.pipeline import load_feature_tables
from src.scoring import scale_p5_p95
from src.tuning import search_best_thresholds
from src.utils import save_json, to_markdown_table


def evaluate_variant(name, sim_col, train_feat, val_feat, test_feat):
    print(f"\n--- Đang đánh giá biến thể: {name} ---")
    actual_col = sim_col if sim_col in train_feat.columns else ("sim_chunk" if "sim_chunk" in train_feat.columns else "sim_trunc")
    percentile_p5, percentile_p95 = (float(v) for v in np.percentile(train_feat[actual_col], [5, 95]))
    print(f"Phân vị Train: p5={percentile_p5:.4f}, p95={percentile_p95:.4f}")

    val_sims = val_feat[actual_col].to_numpy()
    val_scores = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in val_sims], dtype=np.float32)
    y_val = val_feat["label"].to_numpy()

    best_t1, best_t2, val_f1 = search_best_thresholds(val_scores, y_val)
    print(f"Ngưỡng tối ưu: t1={best_t1:.1f}, t2={best_t2:.1f} (Macro-F1 Val: {val_f1:.4f})")

    test_sims = test_feat[actual_col].to_numpy()
    test_scores = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in test_sims], dtype=np.float32)
    y_test = test_feat["label"].to_numpy()

    test_preds = np.zeros_like(test_scores, dtype=int)
    test_preds[test_scores >= best_t1] = 1
    test_preds[test_scores >= best_t2] = 2

    metrics = evaluate_all(
        y_true=y_test,
        y_pred=test_preds,
        continuous_scores=test_scores,
        job_ids=test_feat["job_id"].to_numpy(),
    )
    return {
        "name": name,
        "thresholds": (best_t1, best_t2),
        "scaler": (percentile_p5, percentile_p95),
        "metrics": metrics,
    }


def run_e3():
    print("=== CHẠY THỰC NGHIỆM E3: PRETRAINED MINILM (TRUNCATE VS CHUNKING) ===")
    encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    tables = load_feature_tables(PROJECT_ROOT / "artifacts" / "features" / "minilm_pretrained", encoder)

    col_trunc = "sim_truncated_256" if "sim_truncated_256" in tables["train"].columns else "sim_trunc"
    col_chunk = "sim_chunk_200" if "sim_chunk_200" in tables["train"].columns else "sim_chunk"

    res_3a = evaluate_variant("E3a (Truncate 256)", col_trunc, tables["train"], tables["val"], tables["test"])
    res_3b = evaluate_variant("E3b (Chunking 200/50, JD + CV)", col_chunk, tables["train"], tables["val"], tables["test"])

    headers = ["Mô hình", "Ngưỡng (t1, t2)", "QWK", "Macro-F1", "MAE", "Spearman", "NDCG@10"]
    rows = []
    for r in [res_3a, res_3b]:
        m = r["metrics"]
        t1, t2 = r["thresholds"]
        rows.append([
            r["name"],
            f"({t1:.0f}, {t2:.0f})",
            f"{m['qwk']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{m['mae']:.4f}",
            f"{m.get('spearman', 0.0):.4f}",
            f"{m.get('ndcg_10', 0.0):.4f}",
        ])

    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir = PROJECT_ROOT / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "e3.json", {"e3a": res_3a, "e3b": res_3b})
    with open(out_dir / "e3.md", "w", encoding="utf-8") as f:
        f.write("# Kết quả Thực nghiệm E3: Pretrained MiniLM\n\n" + md_table + "\n")
    print(f"Đã lưu kết quả vào {out_dir}/e3.json và e3.md")


if __name__ == "__main__":
    run_e3()


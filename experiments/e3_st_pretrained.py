import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL, FEATURES_DIR, MAX_SEQ_LENGTH, RESULTS_DIR
from src.metrics import compute_all_metrics
from src.pipeline import get_split_features
from src.scoring import scale_p5_p95
from src.tuning import search_best_thresholds, to_labels
from src.utils import save_json, save_preds, to_markdown_table


def evaluate_variant(name, sim_col, train_feat, val_feat, test_feat):
    percentile_p5, percentile_p95 = (float(v) for v in np.percentile(train_feat[sim_col], [5, 95]))

    val_sims = val_feat[sim_col].to_numpy()
    val_scores = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in val_sims], dtype=np.float32)
    y_val = val_feat["label"].to_numpy()

    best_t1, best_t2, val_f1 = search_best_thresholds(val_scores, y_val)

    test_sims = test_feat[sim_col].to_numpy()
    test_scores = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in test_sims], dtype=np.float32)
    y_test = test_feat["label"].to_numpy()

    test_preds = to_labels(test_scores, best_t1, best_t2)

    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=test_preds,
        continuous_scores=test_scores,
        job_ids=test_feat["job_id"].to_numpy(),
    )
    return {
        "name": name,
        "thresholds": (best_t1, best_t2),
        "scaler": (percentile_p5, percentile_p95),
        "val_macro_f1": val_f1,
        "metrics": metrics,
    }, test_scores, test_preds


def run_e3(seed: int, scheme: str = "jd") -> dict:
    print(f"=== CHẠY THỰC NGHIỆM E3: PRETRAINED MINILM (Seed {seed}, chia {scheme}) ===")
    encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    tables = get_split_features(FEATURES_DIR / "minilm_pretrained", encoder, seed, scheme)
    out_dir = RESULTS_DIR / scheme / f"seed{seed}"

    res_3a, s_3a, p_3a = evaluate_variant("E3a (Truncate 256)", "sim_truncated_256", tables["train"], tables["val"], tables["test"])
    res_3b, s_3b, p_3b = evaluate_variant("E3b (Chunking 200/50, JD + CV)", "sim_chunk_200", tables["train"], tables["val"], tables["test"])
    save_preds(out_dir, "M3a", tables["test"], s_3a, p_3a)
    save_preds(out_dir, "M3b", tables["test"], s_3b, p_3b)

    headers = ["Mô hình", "Ngưỡng (t1, t2)", "QWK", "Macro-F1", "MAE", "Spearman", "NDCG@10"]
    rows = []
    for r in [res_3a, res_3b]:
        m = r["metrics"]
        t1, t2 = r["thresholds"]
        rows.append([
            f"{r['name']} (seed {seed})",
            f"({t1:.0f}, {t2:.0f})",
            f"{m['qwk']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{m['mae']:.4f}",
            f"{m.get('spearman', 0.0):.4f}",
            f"{m.get('ndcg_10', 0.0):.4f}",
        ])

    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"seed": seed, "scheme": scheme, "e3a": res_3a, "e3b": res_3b}
    save_json(out_dir / "e3.json", res)
    with open(out_dir / "e3.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E3: Pretrained MiniLM (Seed {seed})\n\n" + md_table + "\n")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed phân chia dữ liệu.")
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd", help="Cách chia dữ liệu.")
    args = parser.parse_args()
    run_e3(args.seed, args.scheme)

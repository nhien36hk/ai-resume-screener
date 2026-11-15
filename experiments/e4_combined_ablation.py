from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL, MAX_SEQ_LENGTH
from src.pipeline import load_feature_tables
from src.tuning import CRITERIA, evaluate_combined_variant, tune_combined_model
from src.utils import save_json, to_markdown_table


def run_e4():
    print("=== CHẠY THỰC NGHIỆM E4: MÔ HÌNH KẾT HỢP M4 VÀ ABLATION STUDY ===")
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    tables = load_feature_tables(PROJECT_ROOT / "artifacts" / "features" / "minilm_pretrained", model)
    df_val, df_test = tables["val"], tables["test"]

    col_chunk = "sim_chunk_200" if "sim_chunk_200" in tables["train"].columns else "sim_chunk"
    percentile_p5, percentile_p95 = (float(v) for v in np.percentile(tables["train"][col_chunk], [5, 95]))
    print(f"Phân vị Train: p5={percentile_p5:.4f}, p95={percentile_p95:.4f}")

    print("\n1. Tìm skill_threshold, trọng số weights và ngưỡng (t1, t2) trên tập Val...")
    cfg = tune_combined_model(df_val, percentile_p5, percentile_p95)
    weights = cfg["weights"]
    print(f"-> Chọn skill_threshold={cfg['skill_threshold']}, w={dict(zip(CRITERIA, np.round(weights, 2).tolist()))}, t=({cfg['t1']:.0f}, {cfg['t2']:.0f})")

    def without(name):
        return np.where(np.array(CRITERIA) == name, 0.0, weights)

    # D3: giữ w* của bản full, bỏ tiêu chí (mẫu số tự chuẩn hoá lại), tìm lại t1,t2 trên Val.
    # Bỏ tiêu chí nào thì tắt luôn điều kiện must-have của tiêu chí đó.
    variants = [
        ("M4: Full Model", weights, True, True),
        ("Ablation: -Skill", without("skill"), False, True),
        ("Ablation: -Exp", without("exp"), True, False),
        ("Ablation: -Must-have", weights, False, False),
    ]

    print("\n2. Đánh giá Full M4 và các biến thể Ablation trên tập Test...")
    results = []
    for name, w_variant, skill_rule, exp_rule in variants:
        m, (t1, t2) = evaluate_combined_variant(df_val, df_test, percentile_p5, percentile_p95, cfg["skill_threshold"], w_variant, skill_rule, exp_rule)
        results.append({"name": name, "thresholds": (t1, t2), **m})

    full = results[0]
    headers = ["Biến thể", "Ngưỡng (t1, t2)", "QWK", "Δ QWK", "Macro-F1", "Δ F1", "MAE", "Spearman", "NDCG@10", "NDCG@10 (liên tục)"]
    rows = []
    for i, r in enumerate(results):
        d_qwk = "-" if i == 0 else f"{r['qwk'] - full['qwk']:+.4f}"
        d_f1 = "-" if i == 0 else f"{r['macro_f1'] - full['macro_f1']:+.4f}"
        rows.append([
            r["name"], f"({r['thresholds'][0]:.0f}, {r['thresholds'][1]:.0f})",
            f"{r['qwk']:.4f}", d_qwk, f"{r['macro_f1']:.4f}", d_f1,
            f"{r['mae']:.4f}", f"{r['spearman']:.4f}", f"{r['ndcg_10']:.4f}", f"{r['ndcg_10_cont']:.4f}",
        ])
    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir = PROJECT_ROOT / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {**cfg, "weights": dict(zip(CRITERIA, weights.tolist())), "scaler": (percentile_p5, percentile_p95)}
    save_json(out_dir / "e4.json", {"config": config, "results": results})
    with open(out_dir / "e4.md", "w", encoding="utf-8") as f:
        f.write("# Kết quả Thực nghiệm E4: Mô hình kết hợp M4 và Ablation Study\n\n" + md_table + "\n")
    print(f"Đã lưu kết quả vào {out_dir}/e4.json và e4.md")


if __name__ == "__main__":
    run_e4()

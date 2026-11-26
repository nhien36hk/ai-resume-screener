import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DEFAULT_EMBEDDING_MODEL, FEATURES_DIR, MAX_SEQ_LENGTH, RESULTS_DIR
from src.pipeline import get_split_features
from src.metrics import compute_all_metrics, compute_macro_f1, compute_qwk
from src.tuning import (CRITERIA, find_best_hyperparams, fit_ordinal_logit, ordinal_logit_inputs,
                        predict_ordinal_logit, tune_and_test_variant)
from src.utils import save_json, save_preds, to_markdown_table


# Trọng số khởi điểm của tiểu luận (thứ tự CRITERIA: sem, skill, exp, edu, cert), để so với w* tìm trên Val
INITIAL_WEIGHTS = np.array([0.20, 0.30, 0.25, 0.15, 0.10])


def run_e4(seed: int, scheme: str = "jd") -> dict:
    print(f"=== CHẠY THỰC NGHIỆM E4: MÔ HÌNH KẾT HỢP M4 VÀ ABLATION STUDY (Seed {seed}, chia {scheme}) ===")
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    tables = get_split_features(FEATURES_DIR / "minilm_pretrained", model, seed, scheme)
    df_train, df_val, df_test = tables["train"], tables["val"], tables["test"]
    out_dir = RESULTS_DIR / scheme / f"seed{seed}"

    percentile_p5, percentile_p95 = (float(v) for v in np.percentile(df_train["sim_chunk_200"], [5, 95]))
    print(f"Phân vị Train (seed {seed}): p5={percentile_p5:.4f}, p95={percentile_p95:.4f}")

    print(f"\n1. Tìm skill_threshold, trọng số weights và ngưỡng (t1, t2) trên tập Val (seed {seed})...")
    cfg = find_best_hyperparams(df_val, percentile_p5, percentile_p95)
    weights = cfg["weights"]
    must_ratio = cfg.get("must_ratio", 1.0)
    tau = cfg["skill_threshold"]
    print(f"-> Chọn skill_threshold={tau}, must_ratio={must_ratio}, w={dict(zip(CRITERIA, np.round(weights, 2).tolist()))}, t=({cfg['t1']:.0f}, {cfg['t2']:.0f})")

    def without(name):
        return np.where(np.array(CRITERIA) == name, 0.0, weights)

    # Mỗi biến thể bỏ/thay đúng một thành phần; trọng số và (t1, t2) được chọn lại trên Val như M4 (trừ M4, M4_w0).
    # Bỏ tiêu chí nào thì tắt luôn điều kiện must-have của tiêu chí đó.
    # (mã lưu dự đoán, tên, trọng số, luật skill, luật exp, must_ratio, chuẩn hoá lại mẫu số)
    variants = [
        ("M4", "M4: Full Model", weights, True, True, must_ratio, True),
        ("M4_-skill", "Ablation: -Kỹ năng", without("skill"), False, True, must_ratio, True),
        ("M4_-exp", "Ablation: -Kinh nghiệm", without("exp"), True, False, must_ratio, True),
        ("M4_-edu", "Ablation: -Học vấn", without("edu"), True, True, must_ratio, True),
        ("M4_-cert", "Ablation: -Chứng chỉ", without("cert"), True, True, must_ratio, True),
        ("M4_-sem", "Ablation: -Ngữ nghĩa", without("sem"), True, True, must_ratio, True),
        ("M4_-must", "Ablation: -Must-have", weights, False, False, must_ratio, True),
        ("M4_norenorm", "Không chuẩn hoá lại trọng số", weights, True, True, must_ratio, False),
        ("M4_must1", "Luật must gốc (thiếu ≥1 must, ρ=1)", weights, True, True, 1.0, True),
        ("M4_w0", "Trọng số khởi điểm (0.20/0.30/0.25/0.15/0.10)", INITIAL_WEIGHTS, True, True, must_ratio, True),
    ]

    print(f"\n2. Đánh giá Full M4 và các biến thể Ablation trên tập Test (seed {seed})...")
    results = []
    for code, name, w_variant, skill_rule, exp_rule, ratio, renorm in variants:
        m, (t1, t2), final, preds = tune_and_test_variant(
            df_val, df_test, percentile_p5, percentile_p95, tau, w_variant, skill_rule, exp_rule,
            must_ratio=ratio, renormalize=renorm, retune_weights=code not in ("M4", "M4_w0"),
        )
        save_preds(out_dir, code, df_test, final, preds)
        results.append({"name": f"{name} (seed {seed})", "variant": name, "code": code, "thresholds": (t1, t2), **m})

    print(f"\n3. Đối chứng: hồi quy logistic thứ tự học trên Train (seed {seed})...")
    X_train, fill = ordinal_logit_inputs(df_train, tau, percentile_p5, percentile_p95)
    X_val, _ = ordinal_logit_inputs(df_val, tau, percentile_p5, percentile_p95, fill)
    X_test, _ = ordinal_logit_inputs(df_test, tau, percentile_p5, percentile_p95, fill)
    beta, theta = fit_ordinal_logit(X_train, df_train["label"].to_numpy())
    test_preds = predict_ordinal_logit(X_test, beta, theta)
    test_scores = X_test @ beta
    save_preds(out_dir, "M4_ologit", df_test, test_scores, test_preds)
    m = compute_all_metrics(df_test["label"].to_numpy(), test_preds, test_scores, df_test["job_id"].to_numpy())
    m["val_qwk"] = compute_qwk(df_val["label"].to_numpy(), predict_ordinal_logit(X_val, beta, theta))
    m["val_macro_f1"] = compute_macro_f1(df_val["label"].to_numpy(), predict_ordinal_logit(X_val, beta, theta))
    beta_pos = np.clip(beta, 0.0, None)
    ologit = {
        "beta": dict(zip(CRITERIA, beta.tolist())),
        # hệ số âm cắt về 0 rồi chia tổng, để đặt cạnh w* (cùng thang tổng = 1)
        "weights_normalized": dict(zip(CRITERIA, (beta_pos / beta_pos.sum() if beta_pos.sum() > 0 else beta_pos).tolist())),
        "theta": theta.tolist(),
    }
    print(f"-> β={np.round(beta, 3).tolist()}  trọng số chuẩn hoá={np.round(list(ologit['weights_normalized'].values()), 2).tolist()}")
    results.append({"name": f"Hồi quy logistic thứ tự (seed {seed})", "variant": "Hồi quy logistic thứ tự",
                    "code": "M4_ologit", "thresholds": None, **m})  # không dùng ngưỡng: lấy lớp xác suất cao nhất

    full = results[0]
    headers = ["Biến thể", "Ngưỡng (t1, t2)", "QWK Val", "QWK", "Δ QWK", "Macro-F1", "Δ F1", "Accuracy", "MAE",
               "Spearman/JD", "NDCG@10", "Thiếu must vẫn Good Fit"]
    rows = []
    for i, r in enumerate(results):
        d_qwk = "-" if i == 0 else f"{r['qwk'] - full['qwk']:+.4f}"
        d_f1 = "-" if i == 0 else f"{r['macro_f1'] - full['macro_f1']:+.4f}"
        rows.append([
            r["name"], "-" if r["thresholds"] is None else f"({r['thresholds'][0]:.0f}, {r['thresholds'][1]:.0f})",
            f"{r['val_qwk']:.4f}", f"{r['qwk']:.4f}", d_qwk, f"{r['macro_f1']:.4f}", d_f1, f"{r['accuracy']:.4f}",
            f"{r['mae']:.4f}", f"{r['spearman_jd']:.4f}", f"{r['ndcg_10']:.4f}", str(r.get("n_miss_must_pred2", "-")),
        ])
    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    config = {**cfg, "weights": dict(zip(CRITERIA, weights.tolist())), "scaler": (percentile_p5, percentile_p95)}
    res = {"seed": seed, "scheme": scheme, "config": config, "ordinal_logit": ologit, "results": results}
    save_json(out_dir / "e4.json", res)
    with open(out_dir / "e4.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E4: Mô hình kết hợp M4 và Ablation Study (Seed {seed}, chia {scheme})\n\n" + md_table + "\n")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed phân chia dữ liệu.")
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd", help="Cách chia dữ liệu.")
    args = parser.parse_args()
    run_e4(args.seed, args.scheme)

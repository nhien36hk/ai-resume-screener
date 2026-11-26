"""Phân tích lỗi của M4 hoặc M5 trên tập Test (tiểu luận mục phân tích lỗi).

Đọc preds/<method>.parquet do E4/E5 ghi, ghép điểm từng tiêu chí và thông tin JD/CV.
Ghi RESULTS_DIR/<scheme>/error_analysis_<method>.md.
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import DEFAULT_SEEDS, FEATURES_DIR, RESULTS_DIR
from src.data.loader import load_all
from src.metrics import compute_macro_f1, compute_qwk
from src.scoring import scale_p5_p95
from src.utils import load_json, to_markdown_table

N_OFF_BY_ONE = 30


def load_errors_table(method: str, scheme: str, seeds) -> pd.DataFrame:
    meta = load_all()[["pair_id", "job_description_text", "resume_text", "jd_entities", "cv_entities"]]
    frames = []
    for seed in seeds:
        s_dir = RESULTS_DIR / scheme / f"seed{seed}"
        preds_path = s_dir / "preds" / f"{method}.parquet"
        if not preds_path.exists():
            print(f"Bỏ qua seed {seed}: thiếu {preds_path}")
            continue
        cfg = load_json(s_dir / ("e5.json" if method.startswith("M5") else "e4.json"), {})["config"]
        feat_path = (FEATURES_DIR / "minilm_finetuned" / scheme / f"seed{seed}" / "all.parquet" if method.startswith("M5")
                     else FEATURES_DIR / "minilm_pretrained" / "all.parquet")
        tau = cfg["skill_threshold"]
        feat = pd.read_parquet(feat_path)
        feat = feat.assign(
            S_sem=[scale_p5_p95(s, *cfg["scaler"]) for s in feat["sim_chunk_200"]],
            S_skill=feat[f"score_skill_{tau}"], must_missing=feat[f"miss_count_{tau}"],
        )[["pair_id", "S_sem", "S_skill", "score_experience", "score_education", "score_certificate", "must_missing", "total_must_skills"]]
        frames.append(pd.read_parquet(preds_path).merge(feat, on="pair_id").merge(meta, on="pair_id").assign(seed=seed))
    if not frames:
        raise FileNotFoundError(f"Không có file dự đoán {method} nào trong {RESULTS_DIR / scheme}")
    df = pd.concat(frames, ignore_index=True)
    df["error"] = (df["pred"] - df["label"]).abs()
    df["cv_words"] = df["resume_text"].str.split().str.len()
    df["jd_title"] = df["jd_entities"].map(lambda e: e.get("job_title", ""))
    df["cv_title"] = df["cv_entities"].map(lambda e: e.get("candidate_title", ""))
    df["jd_major_group"] = df["jd_entities"].map(lambda e: e.get("major_group", "unknown"))
    return df


def fmt(v):
    return "-" if pd.isna(v) else f"{v:.0f}"


def case_table(rows: pd.DataFrame) -> str:
    return to_markdown_table(
        ["Seed", "pair_id", "Nhãn", "Dự đoán", "S_final", "S_sem", "S_skill", "S_exp", "S_edu", "S_cert", "Must thiếu", "JD", "CV"],
        [[str(r.seed), r.pair_id, str(r.label), str(r.pred), fmt(r.score), fmt(r.S_sem), fmt(r.S_skill), fmt(r.score_experience),
          fmt(r.score_education), fmt(r.score_certificate), f"{int(r.must_missing)}/{int(r.total_must_skills)}",
          str(r.jd_title)[:40], str(r.cv_title)[:40]] for r in rows.itertuples()])


def group_table(df: pd.DataFrame, key: str, title: str) -> str:
    rows = []
    for name, g in df.groupby(key, observed=True):
        rows.append([str(name), str(len(g)), f"{(g['pred'] == g['label']).mean():.3f}",
                     f"{compute_macro_f1(g['label'], g['pred']):.3f}",
                     f"{compute_qwk(g['label'], g['pred']):.3f}" if g["label"].nunique() > 1 else "-",
                     f"{(g['error'] >= 2).mean() * 100:.1f}%"])
    return to_markdown_table([title, "Số cặp", "Accuracy", "Macro-F1", "QWK", "% lệch 2 mức"], rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="M4", help="Mã mô hình trong preds/, vd M4, M5, M4_-must.")
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    df = load_errors_table(args.method, args.scheme, args.seeds)
    df["cv_length_group"] = pd.qcut(df["cv_words"], 3, labels=["ngắn", "trung bình", "dài"])
    big = df[df["error"] >= 2].sort_values(["seed", "label"])
    small = df[df["error"] == 1]
    small = small.sample(min(N_OFF_BY_ONE, len(small)), random_state=0).sort_values(["seed", "label"])
    cut = df["cv_words"].quantile([1 / 3, 2 / 3]).round().astype(int).tolist()

    md = (f"# Phân tích lỗi {args.method} (chia {args.scheme}, seed {sorted(df['seed'].unique().tolist())})\n\n"
          f"Tổng {len(df)} cặp Test; đúng {(df['error'] == 0).mean() * 100:.1f}%, lệch 1 mức {(df['error'] == 1).mean() * 100:.1f}%, "
          f"lệch 2 mức {(df['error'] >= 2).mean() * 100:.1f}% ({len(big)} cặp).\n\n"
          f"## Theo độ dài CV (ngưỡng phân vị: {cut[0]} và {cut[1]} từ)\n\n" + group_table(df, "cv_length_group", "Độ dài CV")
          + "\n\n## Theo nhóm ngành của JD (major_group)\n\n" + group_table(df, "jd_major_group", "major_group")
          + f"\n\n## Toàn bộ cặp lệch 2 mức ({len(big)})\n\n" + case_table(big)
          + f"\n\n## {len(small)} cặp lệch 1 mức chọn ngẫu nhiên (random_state=0)\n\n" + case_table(small) + "\n")
    out = RESULTS_DIR / args.scheme / f"error_analysis_{args.method}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md[:3000])
    print(f"\n-> Đã lưu {out}")


if __name__ == "__main__":
    main()

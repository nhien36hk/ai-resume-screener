"""Tự sinh toàn bộ bảng số liệu cho luận văn bản 20/11 từ JSON/Parquet và tự kiểm toán toán học."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DEFAULT_SEEDS, FEATURES_DIR, ID_TO_LABEL, RESULTS_DIR, SPLITS_DIR
from src.data.loader import load_all
from src.metrics import compute_all_metrics
from src.tuning import CRITERIA, apply_must_have_penalty, prepare_criteria_inputs, to_labels, weighted_sum_scores
from src.utils import load_json, save_json, to_markdown_table

TOL = 1e-6
SEEDS = list(DEFAULT_SEEDS)

E1_E3_SPECS = [
    ("M1", "e1", None, "M1 – so khớp từ khóa"),
    ("M2", "e2", None, "M2 – TF-IDF"),
    ("M3a", "e3", "e3a", "M3a – MiniLM, cắt 256 token"),
    ("M3b", "e3", "e3b", "M3b – MiniLM, chia đoạn"),
]

E4_VARIANTS = [
    ("M4", "M4", "Đầy đủ"),
    ("M4_-skill", "M4_-skill", "Bỏ tiêu chí kỹ năng"),
    ("M4_-exp", "M4_-exp", "Bỏ tiêu chí kinh nghiệm"),
    ("M4_-edu", "M4_-edu", "Bỏ tiêu chí học vấn"),
    ("M4_-cert", "M4_-cert", "Bỏ tiêu chí chứng chỉ"),
    ("M4_-sem", "M4_-sem", "Bỏ tiêu chí ngữ nghĩa"),
    ("M4_-must", "M4_-must", "Bỏ luật điều kiện bắt buộc"),
    ("M4_norenorm", "M4_norenorm", "Không chuẩn hóa lại trọng số"),
    ("M4_must1", "M4_must1", "Luật gốc (ρ = 1)"),
    ("M4_w0", "M4_w0", "Giữ bộ trọng số khởi điểm"),
    ("M4_ologit", "M4_ologit", "Hồi quy logistic thứ tự thay hai ngưỡng"),
]

E5_SPECS = [
    ("M4", "M4 (MiniLM gốc)"),
    ("M5", "M5 (tinh chỉnh)"),
    ("M5_-skill", "M5_-skill (bỏ kỹ năng)"),
    ("M5_-exp", "M5_-exp (bỏ kinh nghiệm)"),
    ("M5_-must", "M5_-must (bỏ luật điều kiện bắt buộc)"),
]

STRICT_MODELS = ["M2", "M3b", "M4", "M5"]

CONFUSION_ROWS = [
    (2, 1, "Good Fit → Potential Fit"),
    (1, 2, "Potential Fit → Good Fit"),
    (1, 0, "Potential Fit → No Fit"),
    (0, 1, "No Fit → Potential Fit"),
    (2, 0, "Good Fit → No Fit"),
    (0, 2, "No Fit → Good Fit"),
]


def fmt_vi(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")


def stat_summary(vals: list[float], decimals: int = 4, scale: float = 1.0, suffix: str = "") -> dict:
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    text = f"{fmt_vi(mean * scale, decimals)}{suffix} ± {fmt_vi(sd * scale, decimals)}{suffix}"
    return {"mean": mean, "sd": sd, "values": [float(v) for v in arr], "text": text}


def verify_preds_against_json(scheme: str, seed: int, code: str, json_metrics: dict) -> pd.DataFrame:
    """Tính lại độc lập mọi chỉ số từ preds/*.parquet và đối chiếu khớp với JSON."""
    pred_path = RESULTS_DIR / scheme / f"seed{seed}" / "preds" / f"{code}.parquet"
    df = pd.read_parquet(pred_path)
    recomp = compute_all_metrics(df["label"], df["pred"], df["score"], df["job_id"])
    for key in ("accuracy", "qwk", "macro_f1", "mae", "spearman", "spearman_jd", "ndcg_10"):
        assert abs(recomp[key] - json_metrics[key]) < TOL, (
            f"{scheme}/seed{seed}/{code} lệch {key}: recomputed={recomp[key]} vs json={json_metrics[key]}"
        )
    assert np.array_equal(recomp["confusion"], json_metrics["confusion"]), (
        f"{scheme}/seed{seed}/{code} lệch ma trận nhầm lẫn"
    )
    return df


def compute_miss_must_stats(pred_df: pd.DataFrame, feat_df: pd.DataFrame, tau: float) -> tuple[int, int, float]:
    merged = pred_df.merge(feat_df[["pair_id", f"miss_count_{tau}"]], on="pair_id", how="inner")
    assert len(merged) == len(pred_df), "Thiếu pair_id khi ghép bảng đặc trưng"
    miss_must = merged[f"miss_count_{tau}"].to_numpy() >= 1
    n_miss_pred2 = int(((merged["pred"].to_numpy() == 2) & miss_must).sum())
    n_miss_total = int(miss_must.sum())
    rate = float(n_miss_pred2 / max(n_miss_total, 1))
    return n_miss_pred2, n_miss_total, rate


def mask_pii(text: str) -> str:
    masked = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    masked = re.sub(r"https?://\S+|www\.\S+", "[URL]", masked)
    masked = re.sub(r"(\+?\d[\d\s().-]{7,}\d)", "[SĐT]", masked)
    return masked


def first_lines(text: str, n_lines: int = 3, max_chars: int = 260) -> str:
    cleaned = mask_pii(text).replace("\r", "")
    parts = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    joined = " / ".join(parts[:n_lines])
    return joined[:max_chars].rstrip() + ("…" if len(joined) > max_chars else "")


def build_table_4_4() -> tuple[list[dict], str]:
    records, rows = [], []
    for seed in SEEDS:
        e4 = load_json(RESULTS_DIR / "jd" / f"seed{seed}" / "e4.json")
        cfg = e4["config"]
        m4 = next(r for r in e4["results"] if r["code"] == "M4")
        w = cfg["weights"]
        assert abs(sum(w.values()) - 1.0) < TOL, f"seed{seed}: tổng trọng số != 1"
        assert cfg["tau"] == cfg["skill_threshold"], f"seed{seed}: tau != skill_threshold"
        assert 0 <= cfg["t1"] < cfg["t2"] <= 100, f"seed{seed}: ngưỡng t1, t2 bất hợp lệ"
        assert [cfg["t1"], cfg["t2"]] == m4["thresholds"], f"seed{seed}: lệch ngưỡng config vs M4"
        for k in CRITERIA:
            assert abs(w[k] - m4["weights"][k]) < 1e-4, f"seed{seed}: lệch trọng số {k}"

        rec = {
            "seed": seed,
            "tau": cfg["tau"],
            "rho": cfg["must_ratio"],
            "weights": {k: float(w[k]) for k in CRITERIA},
            "t1": int(round(cfg["t1"])),
            "t2": int(round(cfg["t2"])),
            "val_qwk": float(cfg["qwk"]),
            "val_f1": float(cfg["f1"]),
        }
        records.append(rec)
        rows.append([
            str(seed),
            fmt_vi(rec["tau"], 1),
            fmt_vi(rec["rho"], 2),
            *[fmt_vi(rec["weights"][k], 2) for k in CRITERIA],
            str(rec["t1"]),
            str(rec["t2"]),
        ])
    headers = ["Seed", "τ", "ρ", "w_sem", "w_skill", "w_exp", "w_edu", "w_cert", "t1", "t2"]
    return records, to_markdown_table(headers, rows)


def build_table_4_5() -> tuple[dict, str, list[float]]:
    metrics_order = ["macro_f1", "qwk", "accuracy", "mae", "spearman", "ndcg_10"]
    collected = {code: {m: [] for m in metrics_order} for code, _, _, _ in E1_E3_SPECS}
    m1_val_spearman = []

    for seed in SEEDS:
        s_dir = RESULTS_DIR / "jd" / f"seed{seed}"
        e1 = load_json(s_dir / "e1.json")
        m1_val_spearman.append(float(e1["val_spearman"]))
        for code, exp_name, sub_key, _ in E1_E3_SPECS:
            data = load_json(s_dir / f"{exp_name}.json")
            m_dict = data[sub_key]["metrics"] if sub_key else data["metrics"]
            verify_preds_against_json("jd", seed, code, m_dict)
            for m in metrics_order:
                collected[code][m].append(float(m_dict[m]))

    summary, rows = {}, []
    for code, _, _, label in E1_E3_SPECS:
        summary[code] = {m: stat_summary(collected[code][m]) for m in metrics_order}
        rows.append([label] + [summary[code][m]["text"] for m in metrics_order])

    headers = ["Mô hình", "Macro-F1", "QWK", "Accuracy", "MAE", "Spearman", "NDCG@10"]
    return summary, to_markdown_table(headers, rows), m1_val_spearman


def build_table_4_6(feat_pretrained: pd.DataFrame) -> tuple[dict, str]:
    metrics_order = ["val_qwk", "macro_f1", "qwk", "accuracy", "spearman_jd"]
    collected = {
        code: {m: [] for m in metrics_order + ["rate_miss_must_pred2", "n_miss_must_pred2", "n_miss_total"]}
        for code, _, _ in E4_VARIANTS
    }

    for seed in SEEDS:
        e4 = load_json(RESULTS_DIR / "jd" / f"seed{seed}" / "e4.json")
        tau = e4["config"]["skill_threshold"]
        by_code = {r["code"]: r for r in e4["results"]}
        for code, _, _ in E4_VARIANTS:
            rec = by_code[code]
            pred_df = verify_preds_against_json("jd", seed, code, rec)
            n_miss2, n_miss_tot, rate = compute_miss_must_stats(pred_df, feat_pretrained, tau)
            if code != "M4_ologit":
                assert n_miss2 == rec["n_miss_must_pred2"], f"seed{seed}/{code} lệch n_miss_must_pred2"
                assert abs(rate - rec["rate_miss_must_pred2"]) < 1e-12, f"seed{seed}/{code} lệch rate"
            if code == "M4_must1":
                assert n_miss2 == 0, f"seed{seed}: M4_must1 vẫn có {n_miss2} cặp thiếu must được xếp Good Fit"
            if code.startswith("M4_-") and code != "M4_-must":
                crit = code.split("_-")[1]
                assert rec["weights"][crit] == 0.0, f"seed{seed}/{code}: trọng số {crit} != 0"

            for m in metrics_order:
                collected[code][m].append(float(rec[m]))
            collected[code]["rate_miss_must_pred2"].append(rate)
            collected[code]["n_miss_must_pred2"].append(n_miss2)
            collected[code]["n_miss_total"].append(n_miss_tot)

    summary, rows = {}, []
    for code, disp, desc in E4_VARIANTS:
        summary[code] = {m: stat_summary(collected[code][m]) for m in metrics_order}
        summary[code]["rate_miss_must_pred2"] = stat_summary(
            collected[code]["rate_miss_must_pred2"], decimals=1, scale=100.0, suffix="%"
        )
        summary[code]["n_miss_must_pred2_sum"] = int(sum(collected[code]["n_miss_must_pred2"]))
        summary[code]["n_miss_total_sum"] = int(sum(collected[code]["n_miss_total"]))
        rows.append(
            [disp, desc]
            + [summary[code][m]["text"] for m in metrics_order]
            + [summary[code]["rate_miss_must_pred2"]["text"]]
        )

    headers = ["Biến thể", "Thay đổi so với M4", "QWK Val", "Macro-F1", "QWK", "Accuracy", "Spearman/JD", "Thiếu must vẫn Good Fit"]
    return summary, to_markdown_table(headers, rows)


def build_table_4_8() -> tuple[dict, str]:
    metrics_order = ["macro_f1", "qwk", "mae", "spearman", "ndcg_10"]
    collected = {code: {m: [] for m in metrics_order} for code, _ in E5_SPECS}

    for seed in SEEDS:
        s_dir = RESULTS_DIR / "jd" / f"seed{seed}"
        e4 = load_json(s_dir / "e4.json")
        e5 = load_json(s_dir / "e5.json")
        m4_rec = next(r for r in e4["results"] if r["code"] == "M4")
        by_code = {"M4": m4_rec, **{r["code"]: r for r in e5["results"]}}
        for code, _ in E5_SPECS:
            rec = by_code[code]
            verify_preds_against_json("jd", seed, code, rec)
            for m in metrics_order:
                collected[code][m].append(float(rec[m]))

    summary, rows = {}, []
    for code, disp in E5_SPECS:
        summary[code] = {m: stat_summary(collected[code][m]) for m in metrics_order}
        rows.append([disp] + [summary[code][m]["text"] for m in metrics_order])

    headers = ["Mô hình", "Macro-F1", "QWK", "MAE", "Spearman", "NDCG@10"]
    return summary, to_markdown_table(headers, rows)


def build_table_4_7(t45: dict, t46: dict, t48: dict) -> tuple[dict, str, list[dict], str]:
    collected = {scheme: {code: {"macro_f1": [], "qwk": []} for code in STRICT_MODELS} for scheme in ("jd", "strict")}

    for scheme in ("jd", "strict"):
        for seed in SEEDS:
            s_dir = RESULTS_DIR / scheme / f"seed{seed}"
            e2 = load_json(s_dir / "e2.json")
            e3 = load_json(s_dir / "e3.json")
            e4 = load_json(s_dir / "e4.json")
            e5 = load_json(s_dir / "e5.json")
            recs = {
                "M2": e2["metrics"],
                "M3b": e3["e3b"]["metrics"],
                "M4": next(r for r in e4["results"] if r["code"] == "M4"),
                "M5": next(r for r in e5["results"] if r["code"] == "M5"),
            }
            for code in STRICT_MODELS:
                verify_preds_against_json(scheme, seed, code, recs[code])
                collected[scheme][code]["macro_f1"].append(float(recs[code]["macro_f1"]))
                collected[scheme][code]["qwk"].append(float(recs[code]["qwk"]))

    summary, rows = {}, []
    for code in STRICT_MODELS:
        jd_f1 = stat_summary(collected["jd"][code]["macro_f1"])
        st_f1 = stat_summary(collected["strict"][code]["macro_f1"])
        jd_qwk = stat_summary(collected["jd"][code]["qwk"])
        st_qwk = stat_summary(collected["strict"][code]["qwk"])
        ref = t45.get(code) or t46.get(code) or t48.get(code)
        assert abs(jd_f1["mean"] - ref["macro_f1"]["mean"]) < TOL, f"Bảng 4.7 lệch JD Macro-F1 của {code}"
        assert abs(jd_qwk["mean"] - ref["qwk"]["mean"]) < TOL, f"Bảng 4.7 lệch JD QWK của {code}"
        summary[code] = {"jd_macro_f1": jd_f1, "strict_macro_f1": st_f1, "jd_qwk": jd_qwk, "strict_qwk": st_qwk}
        rows.append([code, jd_f1["text"], st_f1["text"], jd_qwk["text"], st_qwk["text"]])

    split_records, split_rows = [], []
    for seed in SEEDS:
        sp = load_json(SPLITS_DIR / "strict" / f"seed{seed}.json")
        s_tr, s_va, s_te = set(sp["train"]), set(sp["val"]), set(sp["test"])
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), f"strict seed{seed} bị giao tập"
        tot = len(sp["train"]) + len(sp["val"]) + len(sp["test"])
        rec = {"seed": seed, "train": len(sp["train"]), "val": len(sp["val"]), "test": len(sp["test"]), "total": tot}
        split_records.append(rec)
        split_rows.append([str(seed), str(rec["train"]), str(rec["val"]), str(rec["test"]), str(tot)])

    md_main = to_markdown_table(["Mô hình", "Macro-F1 (theo JD)", "Macro-F1 (chặt)", "QWK (theo JD)", "QWK (chặt)"], rows)
    md_splits = to_markdown_table(["Seed", "Train", "Val", "Test", "Tổng giữ lại"], split_rows)
    return summary, md_main, split_records, md_splits


def build_table_4_9() -> tuple[dict, str]:
    e6 = load_json(RESULTS_DIR / "e6.json")
    rows, records = [], []
    for rec in e6["benchmarks"]:
        steps = rec["steps_mean_sec"]
        assert abs(sum(steps.values()) - rec["total_mean_sec"]) < TOL, f"E6 n_cv={rec['n_cv']}: tổng các bước != total"
        assert abs(rec["total_mean_sec"] / rec["n_cv"] - rec["per_cv_mean_sec"]) < TOL, f"E6 n_cv={rec['n_cv']}: TB/CV sai"
        records.append(rec)
        rows.append([
            str(rec["n_cv"]),
            fmt_vi(steps["jd"] * 1000, 1),
            fmt_vi(steps["cv_encode"] * 1000, 1),
            fmt_vi(steps["skill_encode"] * 1000, 1),
            fmt_vi(steps["scoring"] * 1000, 1),
            f"{fmt_vi(rec['total_mean_sec'] * 1000, 1)} ± {fmt_vi(rec['total_sd_sec'] * 1000, 1)}",
            fmt_vi(rec["per_cv_mean_sec"] * 1000, 1),
            fmt_vi(rec["per_cv_p95_sec"] * 1000, 1),
        ])
    headers = ["Số CV", "Mã hoá JD (ms)", "Mã hoá đoạn CV (ms)", "Mã hoá kỹ năng (ms)", "Tính điểm (ms)", "Tổng (ms)", "TB mỗi CV (ms)", "p95 mỗi CV (ms)"]
    out = {"device": e6["device"], "batch_size": e6["batch_size"], "n_repeats": e6["n_repeats"], "benchmarks": records}
    return out, to_markdown_table(headers, rows)


def build_table_4_10(feat_pretrained: pd.DataFrame, df_all: pd.DataFrame) -> tuple[dict, str, dict]:
    """Bảng 4.10 + kiểm toán toàn bộ dự đoán M4 5 seed + tìm ví dụ thật từ dữ liệu."""
    meta = df_all[["pair_id", "job_description_text", "resume_text", "jd_entities", "cv_entities"]]
    cm_json_sum = np.zeros((3, 3), dtype=int)
    frames = []

    for seed in SEEDS:
        e4 = load_json(RESULTS_DIR / "jd" / f"seed{seed}" / "e4.json")
        cfg = e4["config"]
        m4_rec = next(r for r in e4["results"] if r["code"] == "M4")
        cm_json_sum += np.asarray(m4_rec["confusion"], dtype=int)

        pred_df = pd.read_parquet(RESULTS_DIR / "jd" / f"seed{seed}" / "preds" / "M4.parquet")
        merged = pred_df.merge(feat_pretrained.drop(columns=["job_id", "label"]), on="pair_id")
        tau, rho = cfg["skill_threshold"], cfg["must_ratio"]
        p5, p95 = cfg["scaler"]
        t1, t2 = cfg["t1"], cfg["t2"]
        w = np.array([cfg["weights"][k] for k in CRITERIA])
        cs, cm, sf, ef = prepare_criteria_inputs(merged, tau, p5, p95, must_ratio=rho)
        raw_s = weighted_sum_scores(cs, cm, w)
        failed = sf | ef
        final = apply_must_have_penalty(raw_s, failed, t2)
        assert np.array_equal(to_labels(final, t1, t2), merged["pred"].to_numpy()), f"seed{seed} tái tính nhãn M4 không khớp"

        merged = merged.assign(
            seed=seed,
            S_sem=cs[:, 0],
            S_skill=np.where(cm[:, 1] > 0, cs[:, 1], np.nan),
            S_exp=np.where(cm[:, 2] > 0, cs[:, 2], np.nan),
            S_edu=np.where(cm[:, 3] > 0, cs[:, 3], np.nan),
            S_cert=np.where(cm[:, 4] > 0, cs[:, 4], np.nan),
            S_raw=raw_s,
            S_final=final,
            t1=t1,
            t2=t2,
            must_missing=merged[f"miss_count_{tau}"],
            skill_fail=sf,
            exp_fail=ef,
            demoted_by_rule=(to_labels(raw_s, t1, t2) == 2) & (merged["pred"] == 1),
        )
        frames.append(merged)

    full = pd.concat(frames, ignore_index=True)
    cm_recomp = np.zeros((3, 3), dtype=int)
    for t, p in zip(full["label"], full["pred"]):
        cm_recomp[int(t), int(p)] += 1
    assert np.array_equal(cm_recomp, cm_json_sum), "Ma trận nhầm lẫn cộng dồn 5 seed của M4 không khớp JSON"

    total_pairs = int(cm_recomp.sum())
    total_correct = int(np.trace(cm_recomp))
    err_df = full[full["label"] != full["pred"]].copy()
    total_errors = len(err_df)
    assert total_pairs == 6029 and total_correct + total_errors == total_pairs

    rows, cell_stats = [], []
    rate_sum = 0.0
    for t_lbl, p_lbl, desc in CONFUSION_ROWS:
        sub = err_df[(err_df["label"] == t_lbl) & (err_df["pred"] == p_lbl)]
        cnt = len(sub)
        assert cnt == int(cm_recomp[t_lbl, p_lbl]), f"Lệch ô ({t_lbl}, {p_lbl})"
        rate = cnt / total_errors * 100.0
        rate_sum += rate
        pct_miss_must = float((sub["must_missing"] >= 1).mean() * 100.0) if cnt else 0.0
        pct_demoted = float(sub["demoted_by_rule"].mean() * 100.0) if cnt else 0.0
        pct_zero_exp = float((sub["cv_years_exp"] == 0).mean() * 100.0) if cnt else 0.0
        pct_sem_hi_skill_lo = float(((sub["S_sem"] >= 60) & (sub["S_skill"].fillna(0) < 30)).mean() * 100.0) if cnt else 0.0
        note = (
            f"Thiếu ≥1 must-skill: {fmt_vi(pct_miss_must, 1)}%; "
            f"S_sem ≥ 60 nhưng S_skill < 30: {fmt_vi(pct_sem_hi_skill_lo, 1)}%; "
            f"Kinh nghiệm CV = 0: {fmt_vi(pct_zero_exp, 1)}%"
        )
        if pct_demoted > 0:
            note += f"; Bị hạ mức bởi luật điều kiện bắt buộc: {int(sub['demoted_by_rule'].sum())} cặp ({fmt_vi(pct_demoted, 1)}%)"
        cell_stats.append({
            "true": t_lbl, "pred": p_lbl, "label": desc, "count": cnt, "pct_errors": rate,
            "pct_miss_must": pct_miss_must, "pct_demoted": pct_demoted,
            "pct_zero_exp": pct_zero_exp, "pct_sem_hi_skill_lo": pct_sem_hi_skill_lo,
        })
        rows.append([desc, str(cnt), f"{fmt_vi(rate, 1)}%", note])

    assert abs(rate_sum - 100.0) < 1e-9, f"Tổng tỷ lệ lỗi != 100%: {rate_sum}"

    # Trích ví dụ thực tế (đã ghép meta để hiển thị trích đoạn sạch PII)
    full_meta = full.merge(meta, on="pair_id", how="left")
    demoted_gf = full_meta[(full_meta["label"] == 2) & full_meta["demoted_by_rule"]].sort_values("must_missing")
    err_02 = full_meta[(full_meta["label"] == 0) & (full_meta["pred"] == 2)].sort_values("S_final", ascending=False)
    err_20 = full_meta[(full_meta["label"] == 2) & (full_meta["pred"] == 0)].sort_values("S_final", ascending=True)

    # Tìm cặp minh họa E2 vs E3b (cùng Good Fit: M2 đoán 0, M3b & M4 cùng đoán 2)
    m2_vs_m3b_rows = []
    for seed in SEEDS:
        p2 = pd.read_parquet(RESULTS_DIR / "jd" / f"seed{seed}" / "preds" / "M2.parquet")
        p3b = pd.read_parquet(RESULTS_DIR / "jd" / f"seed{seed}" / "preds" / "M3b.parquet")
        p4 = pd.read_parquet(RESULTS_DIR / "jd" / f"seed{seed}" / "preds" / "M4.parquet")
        m = p2.merge(p3b, on=["pair_id", "job_id", "label"], suffixes=("_m2", "_m3b")).merge(
            p4[["pair_id", "score", "pred"]].rename(columns={"score": "score_m4", "pred": "pred_m4"}), on="pair_id"
        )
        sub = m[(m["label"] == 2) & (m["pred_m2"] == 0) & (m["pred_m3b"] == 2) & (m["pred_m4"] == 2)].assign(seed=seed)
        if not sub.empty:
            m2_vs_m3b_rows.append(sub)
    m2_vs_m3b = pd.concat(m2_vs_m3b_rows, ignore_index=True).assign(
        gap=lambda d: d["score_m3b"] - d["score_m2"]
    ).sort_values("gap", ascending=False).merge(meta, on="pair_id", how="left")

    def pack_m4_case(row) -> dict:
        return {
            "seed": int(row["seed"]), "pair_id": row["pair_id"],
            "label": int(row["label"]), "pred": int(row["pred"]),
            "jd_title": row["jd_entities"].get("job_title", ""),
            "cv_title": row["cv_entities"].get("candidate_title", ""),
            "S_sem": round(float(row["S_sem"]), 1),
            "S_skill": None if pd.isna(row["S_skill"]) else round(float(row["S_skill"]), 1),
            "S_exp": None if pd.isna(row["S_exp"]) else round(float(row["S_exp"]), 1),
            "S_edu": None if pd.isna(row["S_edu"]) else round(float(row["S_edu"]), 1),
            "S_cert": None if pd.isna(row["S_cert"]) else round(float(row["S_cert"]), 1),
            "S_raw": round(float(row["S_raw"]), 1),
            "S_final": round(float(row["S_final"]), 1),
            "t1": int(row["t1"]), "t2": int(row["t2"]),
            "must_missing": int(row["must_missing"]),
            "total_must": int(row["total_must_skills"]),
            "cv_years_exp": float(row["cv_years_exp"]),
            "jd_min_years_exp": float(row["jd_min_years_exp"]),
            "jd_excerpt": first_lines(row["job_description_text"], 3, 240),
            "cv_excerpt": first_lines(row["resume_text"], 3, 240),
        }

    examples = {
        "demoted_good_fit": [pack_m4_case(r) for _, r in demoted_gf.head(3).iterrows()],
        "false_good_fit_0_to_2": [pack_m4_case(r) for _, r in err_02.head(3).iterrows()],
        "missed_good_fit_2_to_0": [pack_m4_case(r) for _, r in err_20.head(3).iterrows()],
        "m2_low_vs_m3b_high": [
            {
                "seed": int(r["seed"]), "pair_id": r["pair_id"], "label": int(r["label"]),
                "score_m2": round(float(r["score_m2"]), 1), "pred_m2": int(r["pred_m2"]),
                "score_m3b": round(float(r["score_m3b"]), 1), "pred_m3b": int(r["pred_m3b"]),
                "score_m4": round(float(r["score_m4"]), 1), "pred_m4": int(r["pred_m4"]),
                "jd_title": r["jd_entities"].get("job_title", ""),
                "cv_title": r["cv_entities"].get("candidate_title", ""),
                "jd_excerpt": first_lines(r["job_description_text"], 3, 240),
                "cv_excerpt": first_lines(r["resume_text"], 3, 240),
            }
            for _, r in m2_vs_m3b.head(5).iterrows()
        ],
    }

    headers = ["Nhãn thật → dự đoán", "Số cặp", "Tỷ lệ trong số lỗi", "Nguyên nhân thường gặp"]
    summary = {"total_pairs": total_pairs, "total_correct": total_correct, "total_errors": total_errors, "cells": cell_stats}
    return summary, to_markdown_table(headers, rows), examples


def build_table_4_2(df_all: pd.DataFrame) -> tuple[list[dict], str]:
    records, rows = [], []
    for lbl in (0, 1, 2):
        r = df_all[df_all["label"] == lbl].iloc[0]
        rec = {
            "label": lbl,
            "label_name": f"{ID_TO_LABEL[lbl]} ({lbl})",
            "pair_id": r["pair_id"],
            "jd_excerpt": first_lines(r["job_description_text"], 3, 240),
            "cv_excerpt": first_lines(r["resume_text"], 3, 240),
        }
        records.append(rec)
        rows.append([rec["label_name"], rec["jd_excerpt"], rec["cv_excerpt"]])
    return records, to_markdown_table(["Nhãn", "Trích JD", "Trích CV"], rows)


def build_e5_train_summary() -> tuple[dict, str]:
    out = {}
    rows = []
    for scheme in ("jd", "strict"):
        out[scheme] = []
        for seed in SEEDS:
            tl = load_json(ARTIFACTS_DIR / "models" / "e5_finetuned" / scheme / f"seed{seed}" / "train_log.json")
            evals = [h for h in tl["log_history"] if "eval_val_spearman_cosine" in h]
            best = max(evals, key=lambda h: h["eval_val_spearman_cosine"])
            rec = {
                "seed": seed,
                "train_seconds": float(tl["train_seconds"]),
                "stopped_epoch": int(round(evals[-1]["epoch"])),
                "best_epoch": int(round(best["epoch"])),
                "best_val_spearman": float(best["eval_val_spearman_cosine"]),
            }
            out[scheme].append(rec)
            rows.append([
                scheme, str(seed), fmt_vi(rec["train_seconds"], 1),
                str(rec["stopped_epoch"]), str(rec["best_epoch"]), fmt_vi(rec["best_val_spearman"], 4),
            ])
    headers = ["Cách chia", "Seed", "Thời gian (s)", "Epoch dừng", "Best epoch", "Val Spearman cao nhất"]
    return out, to_markdown_table(headers, rows)


def build_bootstrap_summary() -> tuple[dict, str]:
    out, parts = {}, []
    for scheme in ("jd", "strict"):
        bp = load_json(RESULTS_DIR / scheme / "bootstrap.json")
        out[scheme] = bp["results"]
        rows = [
            [
                f"{r['a']} − {r['b']}", r["metric"],
                fmt_vi(r["diff"], 4) if r["diff"] < 0 else "+" + fmt_vi(r["diff"], 4),
                f"[{('+' if r['ci_low'] >= 0 else '')}{fmt_vi(r['ci_low'], 4)}; {('+' if r['ci_high'] >= 0 else '')}{fmt_vi(r['ci_high'], 4)}]",
                "Có" if r["significant"] else "Không",
            ]
            for r in bp["results"]
        ]
        parts.append(f"### Bootstrap 95% ({scheme}, {bp['n_boot']} lần)\n\n" + to_markdown_table(
            ["So sánh", "Chỉ số", "Hiệu số", "KTC 95%", "Khác biệt có ý nghĩa"], rows
        ))
    return out, "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Tổng hợp và kiểm toán toàn bộ bảng số liệu luận văn.")
    parser.parse_args()

    feat_pretrained = pd.read_parquet(FEATURES_DIR / "minilm_pretrained" / "all.parquet")
    df_all = load_all()

    t42, md42 = build_table_4_2(df_all)
    t44, md44 = build_table_4_4()
    t45, md45, m1_val_spearman = build_table_4_5()
    t46, md46 = build_table_4_6(feat_pretrained)
    t48, md48 = build_table_4_8()
    t47, md47, strict_splits, md_strict_splits = build_table_4_7(t45, t46, t48)
    t49, md49 = build_table_4_9()
    t410, md410, examples = build_table_4_10(feat_pretrained, df_all)
    e5_train, md_e5_train = build_e5_train_summary()
    boot, md_boot = build_bootstrap_summary()

    payload = {
        "m1_val_spearman": {
            "values": m1_val_spearman,
            "min": min(m1_val_spearman),
            "max": max(m1_val_spearman),
        },
        "table_4_2": t42,
        "table_4_4": t44,
        "table_4_5": t45,
        "table_4_6": t46,
        "table_4_7": t47,
        "strict_splits": strict_splits,
        "table_4_8": t48,
        "table_4_9": t49,
        "table_4_10": t410,
        "e5_train_log": e5_train,
        "bootstrap": boot,
        "examples": examples,
    }
    save_json(RESULTS_DIR / "thesis_tables.json", payload)

    md_sections = [
        "# Toàn bộ bảng số liệu luận văn (tự sinh & kiểm toán tự động từ JSON/Parquet)\n",
        f"## Mục 3.4.4 (V10): `val_spearman` của M1 qua 5 seed\n- Dao động: **{fmt_vi(min(m1_val_spearman), 4)}** đến **{fmt_vi(max(m1_val_spearman), 4)}** ({', '.join(fmt_vi(v, 4) for v in m1_val_spearman)})\n",
        "## Bảng 4.2. Ví dụ cặp CV–JD theo từng nhãn (đã che PII)\n\n" + md42 + "\n",
        "## Bảng 4.4. Tham số của M4 được chọn trên tập kiểm định theo từng seed\n\n" + md44 + "\n",
        "## Bảng 4.5. Kết quả các phương pháp cơ sở trên tập kiểm tra (Mean ± SD, 5 seed)\n\n" + md45 + "\n",
        "## Bảng 4.6. Kết quả của M4 và các biến thể loại bỏ thành phần trên tập kiểm tra\n\n" + md46 + "\n",
        "## Bảng 4.7. So sánh kết quả giữa chia theo JD và chia chặt (tập kiểm tra, 5 seed)\n\n" + md47 + "\n",
        "### Số cặp giữ lại ở mỗi seed khi chia chặt (mục 4.4.3)\n\n" + md_strict_splits + "\n",
        "## Bảng 4.8. Kết quả của M5 và các biến thể trên tập kiểm tra\n\n" + md48 + "\n",
        "### Nhật ký huấn luyện E5 (dừng sớm patience=2)\n\n" + md_e5_train + "\n",
        f"## Bảng 4.9. Thời gian xử lý theo từng bước ({t49['device']}, batch {t49['batch_size']}, {t49['n_repeats']} lần lặp)\n"
        f"- Số token trung bình của một CV: **{fmt_vi(t49['benchmarks'][-1]['avg_cv_tokens'], 0)}** (100 CV) / **{fmt_vi(t49['benchmarks'][0]['avg_cv_tokens'], 0)}** (1 CV)\n\n"
        + md49 + "\n",
        f"## Bảng 4.10. Các nhóm lỗi chính của M4 trên tập kiểm tra (gộp 5 seed, {t410['total_errors']}/{t410['total_pairs']} cặp sai)\n\n" + md410 + "\n",
        "## Kiểm định Bootstrap theo cặp\n\n" + md_boot + "\n",
    ]
    out_md = RESULTS_DIR / "thesis_tables.md"
    out_md.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"KIỂM TOÁN THÀNH CÔNG 100%! Đã lưu {out_md} và {RESULTS_DIR / 'thesis_tables.json'}")


if __name__ == "__main__":
    main()

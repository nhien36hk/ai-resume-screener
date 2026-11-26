"""Mục 5 spec: grid search trọng số (QWK) và ngưỡng (Macro-F1), dùng chung cho E1–E5."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.config import DEFAULT_THRESHOLDS, MUST_HAVE_RATIOS, SKILL_THRESHOLDS
from src.metrics import compute_all_metrics, compute_macro_f1, compute_qwk
from src.scoring import scale_p5_p95

CRITERIA = ("sem", "skill", "exp", "edu", "cert")


def generate_weight_grid() -> np.ndarray:
    """Tạo đúng 3.246 tổ hợp 5 trọng số: w_i in [0.05, 0.50] bước 0.05, tổng = 1.0."""
    steps = [round(x * 0.05, 2) for x in range(1, 11)]
    combos = []
    for w1 in steps:
        for w2 in steps:
            for w3 in steps:
                for w4 in steps:
                    w5 = round(1.0 - (w1 + w2 + w3 + w4), 2)
                    if 0.05 <= w5 <= 0.50:
                        combos.append([w1, w2, w3, w4, w5])
    return np.asarray(combos, dtype=np.float64)  # float32 in/ghi json ra 0.10000000149


def to_labels(scores: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """S < t1 -> 0, t1 <= S < t2 -> 1, S >= t2 -> 2."""
    return np.where(scores >= t2, 2, np.where(scores >= t1, 1, 0))


def apply_must_have_penalty(scores: np.ndarray, failed: np.ndarray, t2: float) -> np.ndarray:
    """Spec mục 4: cặp vi phạm must-have bị chặn ở t2 - 1."""
    return np.where(failed, np.minimum(scores, t2 - 1.0), scores)


def prepare_criteria_inputs(
    feat: pd.DataFrame,
    skill_threshold: float,
    percentile_p5: float,
    percentile_p95: float,
    sim_col: str = "sim_chunk_200",
    must_ratio: float = 1.0,
):
    """Từ bảng feature: criteria_scores [n×5], criteria_mask [n×5], cờ thiếu must-skill, cờ thiếu kinh nghiệm."""
    sem = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in feat[sim_col]])
    others = feat[[f"score_skill_{skill_threshold}", "score_experience", "score_education", "score_certificate"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    criteria_scores = np.column_stack([sem, others])
    criteria_mask = (~np.isnan(criteria_scores)).astype(float)
    tot = feat["total_must_skills"].to_numpy(float)
    cov = 1.0 - feat[f"miss_count_{skill_threshold}"].to_numpy(float) / np.maximum(tot, 1.0)
    skill_fail = (tot > 0) & (cov < must_ratio)
    return np.nan_to_num(criteria_scores), criteria_mask, skill_fail, feat["has_insufficient_exp"].to_numpy(bool)


def weighted_sum_scores(scores_matrix: np.ndarray, mask_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Tính điểm tổng hợp có trọng số: S = Σ_{i∈A} w_i S_i / Σ_{i∈A} w_i."""
    den = mask_matrix @ weights
    return (scores_matrix * mask_matrix) @ weights / np.where(den > 0, den, 1e-9)


def search_best_weights(
    scores_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    y_true: np.ndarray,
    must_have_failed: Optional[np.ndarray] = None,
    t1: float = DEFAULT_THRESHOLDS[0],
    t2: float = DEFAULT_THRESHOLDS[1],
    weight_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """B1: bộ trọng số cho QWK cao nhất trên Val tại ngưỡng khởi điểm (40, 70). Hoà -> bộ đứng trước."""
    if weight_grid is None:
        weight_grid = generate_weight_grid()
    composite_scores = weighted_sum_scores(scores_matrix, mask_matrix, weight_grid.T)  # (n, 3246)
    if must_have_failed is not None:
        composite_scores = apply_must_have_penalty(composite_scores, must_have_failed[:, None], t2)

    best_qwk, best_weights = -1.0, weight_grid[0]
    for col in range(composite_scores.shape[1]):
        qwk = compute_qwk(y_true, to_labels(composite_scores[:, col], t1, t2))
        if qwk > best_qwk:
            best_qwk, best_weights = qwk, weight_grid[col]
    return best_weights, best_qwk


def search_best_thresholds(
    continuous_scores: np.ndarray,
    y_true: np.ndarray,
    must_have_failed: Optional[np.ndarray] = None,
    min_t1: int = 0,
    max_t2: int = 100,
    step: int = 1,
) -> Tuple[float, float, float]:
    """B2: cặp t1 < t2 (bước 1 trên [0, 100]) cho Macro-F1 cao nhất. Must-have dùng chính t2 đang thử."""
    y = np.asarray(y_true, dtype=int)
    best_f1, best_t1, best_t2 = -1.0, float(DEFAULT_THRESHOLDS[0]), float(DEFAULT_THRESHOLDS[1])
    for t1 in range(min_t1, max_t2, step):
        for t2 in range(t1 + 1, max_t2 + 1, step):
            s = continuous_scores if must_have_failed is None else apply_must_have_penalty(continuous_scores, must_have_failed, t2)
            f1 = compute_macro_f1(y, to_labels(s, t1, t2))
            if f1 > best_f1:
                best_f1, best_t1, best_t2 = f1, float(t1), float(t2)
    return best_t1, best_t2, best_f1


def find_best_hyperparams(
    val_feat: pd.DataFrame,
    percentile_p5: float,
    percentile_p95: float,
    skill_thresholds=SKILL_THRESHOLDS,
    must_ratios=MUST_HAVE_RATIOS,
) -> Dict:
    """Tìm skill_threshold, must_ratio, trọng số weights và ngưỡng (t1, t2) tối ưu trên Val."""
    y = val_feat["label"].to_numpy()
    grid = generate_weight_grid()
    best = None
    for threshold in skill_thresholds:
        for ratio in must_ratios:
            criteria_scores, criteria_mask, skill_fail, exp_fail = prepare_criteria_inputs(
                val_feat, threshold, percentile_p5, percentile_p95, must_ratio=ratio
            )
            failed = skill_fail | exp_fail
            weights, _ = search_best_weights(criteria_scores, criteria_mask, y, failed, weight_grid=grid)
            composite_scores = weighted_sum_scores(criteria_scores, criteria_mask, weights)
            t1, t2, f1 = search_best_thresholds(composite_scores, y, failed)
            qwk = compute_qwk(y, to_labels(apply_must_have_penalty(composite_scores, failed, t2), t1, t2))
            print(f"  threshold={threshold}, must_ratio={ratio}: QWK={qwk:.4f}  Macro-F1={f1:.4f}  weights={np.round(weights, 2).tolist()}  thresholds=({t1:.0f}, {t2:.0f})")
            if best is None or (qwk, f1) > (best["qwk"], best["f1"]):
                best = {
                    "skill_threshold": threshold,
                    "tau": threshold,
                    "must_ratio": ratio,
                    "weights": weights,
                    "t1": t1,
                    "t2": t2,
                    "qwk": qwk,
                    "f1": f1,
                }
    return best


def tune_and_test_variant(
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    percentile_p5: float,
    percentile_p95: float,
    skill_threshold: float,
    weights: np.ndarray,
    skill_rule: bool = True,
    exp_rule: bool = True,
    must_ratio: float = 1.0,
    renormalize: bool = True,
    retune_weights: bool = False,
) -> Tuple[Dict[str, float], Tuple[float, float], np.ndarray, np.ndarray]:
    """Chọn lại tham số trên Val cho biến thể rồi chấm Test một lần.
    Tiêu chí có trọng số 0 trong `weights` là tiêu chí bị bỏ.
    retune_weights=True: dò lại lưới trọng số (QWK) trên các tiêu chí còn lại, như M4 (Bảng 4.6: "các tham số
    còn lại của mỗi biến thể được chọn lại trên tập kiểm định như M4"); False: giữ `weights` (M4, M4_w0).
    skill_rule / exp_rule: bật điều kiện must-have tương ứng (ablation tắt điều kiện của tiêu chí bị bỏ).
    renormalize=False: tiêu chí thiếu thông tin tính 0 điểm, mẫu số giữ nguyên (biến thể ablation Bảng 4.6).
    Trả (metrics Test + val_qwk/val_macro_f1/weights, (t1, t2), điểm S_final Test, nhãn dự đoán Test)."""
    dropped = np.asarray(weights) == 0

    def inputs(feat):
        criteria_scores, criteria_mask, skill_fail, exp_fail = prepare_criteria_inputs(
            feat, skill_threshold, percentile_p5, percentile_p95, must_ratio=must_ratio
        )
        if not renormalize:
            criteria_mask = np.ones_like(criteria_mask)
        criteria_mask[:, dropped] = 0.0
        return criteria_scores, criteria_mask, (skill_fail & skill_rule) | (exp_fail & exp_rule)

    y_val = val_feat["label"].to_numpy()
    val_cs, val_cm, val_failed = inputs(val_feat)
    if retune_weights:
        weights, _ = search_best_weights(val_cs, val_cm, y_val, val_failed)
        weights = np.where(dropped, 0.0, weights)
        weights = weights / weights.sum()
    val_s = weighted_sum_scores(val_cs, val_cm, weights)
    t1, t2, val_f1 = search_best_thresholds(val_s, y_val, val_failed)
    test_cs, test_cm, test_failed = inputs(test_feat)
    test_s = weighted_sum_scores(test_cs, test_cm, weights)
    final = apply_must_have_penalty(test_s, test_failed, t2)
    preds = to_labels(final, t1, t2)
    final = final + 1e-6 * test_s  # V5: CV cùng bị chặn ở t2-1 xếp tiếp theo S trước khi chặn (Spearman, NDCG)
    metrics = compute_all_metrics(test_feat["label"].to_numpy(), preds, final, test_feat["job_id"].to_numpy())
    metrics["val_qwk"] = compute_qwk(y_val, to_labels(apply_must_have_penalty(val_s, val_failed, t2), t1, t2))
    metrics["val_macro_f1"] = val_f1
    metrics["weights"] = dict(zip(CRITERIA, np.round(weights, 4).tolist()))
    # Cặp thiếu >= 1 must-skill mà vẫn bị xếp Good Fit (cột cuối Bảng 4.6: số cặp và tỷ lệ trong số cặp thiếu must)
    miss_must = test_feat[f"miss_count_{skill_threshold}"].to_numpy() >= 1
    metrics["n_miss_must_pred2"] = int(((preds == 2) & miss_must).sum())
    metrics["rate_miss_must_pred2"] = float(((preds == 2) & miss_must).sum() / max(miss_must.sum(), 1))
    return metrics, (t1, t2), final, preds


# ── Đối chứng trọng số: hồi quy logistic thứ tự (tiểu luận 3.6.4) ──────────────

def fit_ordinal_logit(X: np.ndarray, y: np.ndarray, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative logit P(y <= k) = σ(θ_k − X·β), học bằng tối đa likelihood. Trả (β, θ tăng dần)."""
    n, d = X.shape
    y = np.asarray(y, dtype=int)

    def unpack(p):
        return p[:d], np.cumsum(np.r_[p[d], np.exp(p[d + 1:])])  # θ tăng dần qua exp

    def nll(p):
        beta, theta = unpack(p)
        cdf = expit(theta[None, :] - (X @ beta)[:, None])
        cdf = np.hstack([np.zeros((n, 1)), cdf, np.ones((n, 1))])
        prob = cdf[np.arange(n), y + 1] - cdf[np.arange(n), y]
        return -np.log(np.clip(prob, 1e-12, None)).sum()

    res = minimize(nll, np.zeros(d + n_classes - 1), method="L-BFGS-B")
    return unpack(res.x)


def predict_ordinal_logit(X: np.ndarray, beta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Lớp có xác suất cao nhất."""
    cdf = expit(theta[None, :] - (X @ beta)[:, None])
    cdf = np.hstack([np.zeros((len(X), 1)), cdf, np.ones((len(X), 1))])
    return np.argmax(np.diff(cdf, axis=1), axis=1)


def ordinal_logit_inputs(feat: pd.DataFrame, skill_threshold: float, percentile_p5: float, percentile_p95: float,
                         fill: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """5 điểm tiêu chí /100; ô thiếu thông tin điền bằng `fill` (trung bình cột của Train). Trả (X, fill)."""
    scores, mask, _, _ = prepare_criteria_inputs(feat, skill_threshold, percentile_p5, percentile_p95)
    X = np.where(mask > 0, scores / 100.0, np.nan)
    if fill is None:
        fill = np.nan_to_num(np.nanmean(X, axis=0))
    return np.where(np.isnan(X), fill, X), fill

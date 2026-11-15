"""Mục 5 spec: grid search trọng số (QWK) và ngưỡng (Macro-F1), dùng chung cho E1–E5."""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_THRESHOLDS, SKILL_THRESHOLDS
from src.metrics import compute_macro_f1, compute_qwk, evaluate_all
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


def cap_must_have(scores: np.ndarray, failed: np.ndarray, t2: float) -> np.ndarray:
    """Spec mục 4: cặp vi phạm must-have bị chặn ở t2 - 1."""
    return np.where(failed, np.minimum(scores, t2 - 1.0), scores)


def prepare_criteria_inputs(feat: pd.DataFrame, skill_threshold: float, percentile_p5: float, percentile_p95: float, sim_col: str = "sim_chunk_200"):
    """Từ bảng feature: criteria_scores_matrix [n×5], criteria_mask_matrix [n×5], cờ thiếu must-skill, cờ kinh nghiệm < 50%."""
    sim_col_actual = sim_col if sim_col in feat.columns else ("sim_chunk" if "sim_chunk" in feat.columns else sim_col)
    sem = np.array([scale_p5_p95(s, percentile_p5, percentile_p95) for s in feat[sim_col_actual]])
    others = feat[[f"skill_{skill_threshold}", "score_experience", "score_education", "score_certificate"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    criteria_scores_matrix = np.column_stack([sem, others])
    criteria_mask_matrix = (~np.isnan(criteria_scores_matrix)).astype(float)
    skill_fail = feat[f"must_miss_{skill_threshold}"].to_numpy() >= 1
    exp_col = "has_insufficient_exp" if "has_insufficient_exp" in feat.columns else "exp_fail"
    has_insufficient_exp = feat[exp_col].to_numpy(dtype=bool)
    return np.nan_to_num(criteria_scores_matrix), criteria_mask_matrix, skill_fail, has_insufficient_exp


# Backward compatibility alias
m4_inputs = prepare_criteria_inputs


def aggregate(scores_matrix: np.ndarray, mask_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
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
    composite_scores = aggregate(scores_matrix, mask_matrix, weight_grid.T)  # (n, 3246)
    if must_have_failed is not None:
        composite_scores = cap_must_have(composite_scores, must_have_failed[:, None], t2)

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
            s = continuous_scores if must_have_failed is None else cap_must_have(continuous_scores, must_have_failed, t2)
            f1 = compute_macro_f1(y, to_labels(s, t1, t2))
            if f1 > best_f1:
                best_f1, best_t1, best_t2 = f1, float(t1), float(t2)
    return best_t1, best_t2, best_f1


def tune_combined_model(val_feat: pd.DataFrame, percentile_p5: float, percentile_p95: float, skill_thresholds=SKILL_THRESHOLDS) -> Dict:
    """Tìm skill_threshold, bộ trọng số weights và ngưỡng (t1, t2) tối ưu trên tập Validation."""
    y = val_feat["label"].to_numpy()
    grid = generate_weight_grid()
    best = None
    for threshold in skill_thresholds:
        criteria_scores, criteria_mask, skill_fail, exp_fail = prepare_criteria_inputs(val_feat, threshold, percentile_p5, percentile_p95)
        failed = skill_fail | exp_fail
        weights, _ = search_best_weights(criteria_scores, criteria_mask, y, failed, weight_grid=grid)
        composite_scores = aggregate(criteria_scores, criteria_mask, weights)
        t1, t2, f1 = search_best_thresholds(composite_scores, y, failed)
        qwk = compute_qwk(y, to_labels(cap_must_have(composite_scores, failed, t2), t1, t2))
        print(f"  skill_threshold={threshold}: QWK={qwk:.4f}  Macro-F1={f1:.4f}  weights={np.round(weights, 2).tolist()}  thresholds=({t1:.0f}, {t2:.0f})")
        if best is None or (qwk, f1) > (best["qwk"], best["f1"]):
            best = {"skill_threshold": threshold, "tau": threshold, "weights": weights, "t1": t1, "t2": t2, "qwk": qwk, "f1": f1}
    return best


def evaluate_combined_variant(
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    percentile_p5: float,
    percentile_p95: float,
    skill_threshold: float,
    weights: np.ndarray,
    skill_rule: bool = True,
    exp_rule: bool = True,
) -> Tuple[Dict[str, float], Tuple[float, float]]:
    """Tìm lại (t1, t2) trên Val cho biến thể (D3), rồi chấm Test một lần.
    skill_rule / exp_rule: bật điều kiện must-have tương ứng (ablation tắt điều kiện của tiêu chí bị bỏ)."""
    def scores(feat):
        criteria_scores, criteria_mask, skill_fail, exp_fail = prepare_criteria_inputs(feat, skill_threshold, percentile_p5, percentile_p95)
        return aggregate(criteria_scores, criteria_mask, weights), (skill_fail & skill_rule) | (exp_fail & exp_rule)

    val_s, val_failed = scores(val_feat)
    t1, t2, _ = search_best_thresholds(val_s, val_feat["label"].to_numpy(), val_failed)
    test_s, test_failed = scores(test_feat)
    final = cap_must_have(test_s, test_failed, t2)
    metrics = evaluate_all(test_feat["label"].to_numpy(), to_labels(final, t1, t2), final, test_feat["job_id"].to_numpy())
    return metrics, (t1, t2)


# Backward compatibility aliases
tune_m4 = tune_combined_model
evaluate_m4_variant = evaluate_combined_variant

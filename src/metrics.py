from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error, ndcg_score


def compute_qwk(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> float:
    """Quadratic Weighted Kappa (QWK)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_macro_f1(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> float:
    """Macro-F1 score."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_mae(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> float:
    """Mean Absolute Error (MAE)."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_spearman(y_true: Union[List[int], np.ndarray], continuous_scores: Union[List[float], np.ndarray]) -> float:
    """Spearman rank correlation giữa điểm liên tục và nhãn thật."""
    res = spearmanr(y_true, continuous_scores)
    corr = res.statistic if hasattr(res, "statistic") else res.correlation
    return 0.0 if np.isnan(corr) else float(corr)


def compute_ndcg_at_10(
    job_ids: Union[List[str], np.ndarray],
    y_true: Union[List[int], np.ndarray],
    y_pred_scores: Union[List[float], np.ndarray],
    min_cvs: int = 5,
) -> float:
    """NDCG@10 xếp hạng CV theo từng job_id có >= 5 CVs, bỏ qua JD có toàn bộ nhãn 0."""
    df = pd.DataFrame({"job_id": job_ids, "true": y_true, "score": y_pred_scores})
    ndcg_list = []

    for _, group in df.groupby("job_id"):
        if len(group) < min_cvs:
            continue
        rel = group["true"].to_numpy().reshape(1, -1)
        if rel.max() == 0:  # Không có CV phù hợp nào -> IDCG = 0
            continue
        pred = group["score"].to_numpy().reshape(1, -1)
        k = min(10, len(group))
        score = ndcg_score(rel, pred, k=k)
        ndcg_list.append(score)

    return float(np.mean(ndcg_list)) if ndcg_list else 0.0


def evaluate_all(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    continuous_scores: Optional[Union[List[float], np.ndarray]] = None,
    job_ids: Optional[Union[List[str], np.ndarray]] = None,
) -> Dict[str, float]:
    """Tổng hợp toàn bộ chỉ số đánh giá."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    metrics = {
        "qwk": compute_qwk(y_true_arr, y_pred_arr),
        "macro_f1": compute_macro_f1(y_true_arr, y_pred_arr),
        "mae": compute_mae(y_true_arr, y_pred_arr),
    }

    if job_ids is not None:
        metrics["ndcg_10"] = compute_ndcg_at_10(job_ids, y_true_arr, y_pred_arr)  # spec: xếp theo y_pred
    if continuous_scores is not None:
        c_scores = np.asarray(continuous_scores, dtype=np.float32)
        metrics["spearman"] = compute_spearman(y_true_arr, c_scores)
        if job_ids is not None:
            metrics["ndcg_10_cont"] = compute_ndcg_at_10(job_ids, y_true_arr, c_scores)  # D8: xếp theo điểm liên tục

    return metrics


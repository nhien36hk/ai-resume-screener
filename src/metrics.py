from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, ndcg_score


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
    """NDCG@10 theo từng job_id có >= 5 CV, bỏ JD toàn nhãn 0. Gain = 2^rel - 1 (tiểu luận 3.8.9)."""
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
        score = ndcg_score(2 ** rel - 1, pred, k=k)  # sklearn dùng gain tuyến tính -> đổi relevance để ra gain mũ
        ndcg_list.append(score)

    return float(np.mean(ndcg_list)) if ndcg_list else 0.0


def compute_spearman_per_jd(
    job_ids: Union[List[str], np.ndarray],
    y_true: Union[List[int], np.ndarray],
    continuous_scores: Union[List[float], np.ndarray],
    min_cvs: int = 5,
) -> float:
    """Trung bình Spearman trong từng JD (>= 5 CV, nhãn không hằng) — tiểu luận 3.8.9."""
    df = pd.DataFrame({"job_id": job_ids, "true": y_true, "score": continuous_scores})
    vals = [
        compute_spearman(g["true"], g["score"])
        for _, g in df.groupby("job_id")
        if len(g) >= min_cvs and g["true"].nunique() > 1
    ]
    return float(np.mean(vals)) if vals else 0.0


def paired_bootstrap_ci(pairs, metric, n_boot: int = 1000, by_jd: bool = False, rng_seed: int = 0):
    """KTC 95% của hiệu metric(A) - metric(B), bootstrap theo cặp, phân tầng theo seed (tiểu luận 3.8.10).

    pairs: list, mỗi seed một (df_a, df_b) cùng pair_id và cùng thứ tự (cột label, score, pred, job_id).
    Mỗi vòng: lấy mẫu lại trong từng seed (by_jd=True: lấy mẫu JD, cho NDCG), lấy trung bình hiệu số qua seed.
    Trả (hiệu số trên dữ liệu gốc, cận dưới, cận trên).
    """
    rng = np.random.default_rng(rng_seed)

    def draw(df):
        if not by_jd:
            return rng.integers(0, len(df), len(df)), None
        groups = list(df.groupby("job_id").indices.values())  # vị trí dòng của từng JD
        picks = rng.integers(0, len(groups), len(groups))
        pos = np.concatenate([groups[k] for k in picks])
        new_ids = np.concatenate([[f"b{n}"] * len(groups[k]) for n, k in enumerate(picks)])  # JD bốc trùng = JD riêng
        return pos, new_ids

    def diff(samples):
        return float(np.mean([metric(a) - metric(b) for a, b in samples]))

    point = diff(pairs)
    boots = []
    for _ in range(n_boot):
        samples = []
        for a, b in pairs:
            pos, new_ids = draw(a)
            a_s, b_s = a.iloc[pos], b.iloc[pos]
            if new_ids is not None:
                a_s, b_s = a_s.assign(job_id=new_ids), b_s.assign(job_id=new_ids)
            samples.append((a_s, b_s))
        boots.append(diff(samples))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def compute_all_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    continuous_scores: Optional[Union[List[float], np.ndarray]] = None,
    job_ids: Optional[Union[List[str], np.ndarray]] = None,
) -> Dict[str, float]:
    """Tổng hợp toàn bộ chỉ số đánh giá."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "qwk": compute_qwk(y_true_arr, y_pred_arr),
        "macro_f1": compute_macro_f1(y_true_arr, y_pred_arr),
        "mae": compute_mae(y_true_arr, y_pred_arr),
        "confusion": confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2]).tolist(),  # hàng = nhãn thật
    }

    if continuous_scores is not None:
        c_scores = np.asarray(continuous_scores, dtype=np.float64)
        metrics["spearman"] = compute_spearman(y_true_arr, c_scores)
        if job_ids is not None:
            metrics["spearman_jd"] = compute_spearman_per_jd(job_ids, y_true_arr, c_scores)
            metrics["ndcg_10"] = compute_ndcg_at_10(job_ids, y_true_arr, c_scores)  # xếp theo S_final (3.8.9)

    return metrics


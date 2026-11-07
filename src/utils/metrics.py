# Level 1: Thư viện chuẩn Python (Standard Library)
from typing import Dict, Optional, Sequence, Union

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error, ndcg_score

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


MIN_CV_PER_JOB_DEFAULT = 5
TOP_K_NDCG_DEFAULT = 10


def calculate_qwk(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Tính Quadratic Weighted Kappa (QWK) giữa nhãn thực tế và nhãn dự đoán."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def calculate_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """Tính Macro-F1 trung bình không trọng số giữa các lớp."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def calculate_mae(
    y_true: Sequence[Union[int, float]],
    y_pred: Sequence[Union[int, float]],
) -> float:
    """Tính Sai số Tuyệt đối Trung bình (Mean Absolute Error)."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_spearman(
    y_true: Sequence[Union[int, float]],
    y_pred: Sequence[Union[int, float]],
) -> float:
    """Tính hệ số tương quan thứ hạng Spearman."""
    corr = spearmanr(y_true, y_pred).statistic
    return float(corr) if not np.isnan(corr) else 0.0


def calculate_mean_ndcg(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str = "label",
    job_id_col: str = "job_id",
    k: int = TOP_K_NDCG_DEFAULT,
    min_cvs: int = MIN_CV_PER_JOB_DEFAULT,
) -> float:
    """Tính Mean NDCG@k trên tất cả các job_id có số lượng CV >= min_cvs."""
    scores = []
    grouped = df.groupby(job_id_col)

    for _, group in grouped:
        if len(group) < min_cvs:
            continue

        y_true = np.asarray([group[true_col].values])
        y_pred = np.asarray([group[pred_col].values])

        # Nếu tất cả các nhãn thực tế đều bằng nhau (ví dụ toàn 0), NDCG = 1.0 theo định nghĩa
        if np.all(y_true == y_true[0, 0]):
            scores.append(1.0)
            continue

        score = ndcg_score(y_true, y_pred, k=k)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def evaluate_all_metrics(
    y_true: Sequence[int],
    y_pred_labels: Sequence[int],
    y_pred_scores: Optional[Sequence[float]] = None,
    df_eval: Optional[pd.DataFrame] = None,
    pred_score_col: str = "pred_score",
) -> Dict[str, float]:
    """Tính toàn bộ các metrics theo đúng đặc tả Mục 6 trong plan.md."""
    metrics = {
        "QWK": calculate_qwk(y_true, y_pred_labels),
        "Macro_F1": calculate_macro_f1(y_true, y_pred_labels),
        "MAE": calculate_mae(y_true, y_pred_labels),
    }

    continuous_scores = y_pred_scores if y_pred_scores is not None else y_pred_labels
    metrics["Spearman"] = calculate_spearman(y_true, continuous_scores)

    if df_eval is not None and "job_id" in df_eval.columns:
        eval_copy = df_eval.copy()
        eval_copy[pred_score_col] = continuous_scores
        metrics["NDCG@10"] = calculate_mean_ndcg(eval_copy, pred_col=pred_score_col)

    return metrics


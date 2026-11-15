import numpy as np

from src.metrics import compute_macro_f1, compute_ndcg_at_10, compute_qwk


def test_qwk_few_shot_examples():
    # Đúng hoàn toàn -> QWK = 1.0
    y_true = [2, 1, 0, 1]
    y_pred = [2, 1, 0, 1]
    assert compute_qwk(y_true, y_pred) == 1.0

    # Sai 1 bậc (nhẹ) vs Sai 2 bậc (nặng)
    y_target = [2, 2, 0, 0]
    y_pred_1step = [1, 1, 1, 1]  # Sai 1 bậc
    y_pred_2step = [0, 0, 2, 2]  # Sai 2 bậc ngược hoàn toàn

    qwk_1step = compute_qwk(y_target, y_pred_1step)
    qwk_2step = compute_qwk(y_target, y_pred_2step)
    assert qwk_1step > qwk_2step, "Sai 2 bậc phải bị phạt nặng hơn sai 1 bậc"


def test_macro_f1():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    assert compute_macro_f1(y_true, y_pred) == 1.0


def test_ndcg_at_10():
    job_ids = ["JD1"] * 5
    y_true = [2, 1, 0, 0, 0]
    # Sắp xếp đúng hoàn hảo: điểm dự đoán tương quan chuẩn với nhãn thật
    scores_perfect = [90.0, 70.0, 30.0, 20.0, 10.0]
    ndcg_perfect = compute_ndcg_at_10(job_ids, y_true, scores_perfect)
    assert abs(ndcg_perfect - 1.0) < 1e-4

    # Sắp xếp ngược hoàn toàn
    scores_inverted = [10.0, 20.0, 30.0, 70.0, 90.0]
    ndcg_inverted = compute_ndcg_at_10(job_ids, y_true, scores_inverted)
    assert ndcg_perfect > ndcg_inverted



def test_evaluate_all_reports_both_ndcg_columns():
    from src.metrics import evaluate_all
    m = evaluate_all([2, 1, 0, 0, 0], [2, 2, 0, 0, 0], [90, 70, 30, 20, 10], ["J"] * 5)
    assert "ndcg_10" in m and "ndcg_10_cont" in m   # spec (theo y_pred) + D8 (theo điểm liên tục)
    assert abs(m["ndcg_10_cont"] - 1.0) < 1e-6

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



def test_compute_all_metrics_ranks_by_continuous_score():
    from src.metrics import compute_all_metrics
    m = compute_all_metrics([2, 1, 0, 0, 0], [2, 2, 0, 0, 0], [90, 70, 30, 20, 10], ["J"] * 5)
    assert abs(m["ndcg_10"] - 1.0) < 1e-6          # xếp theo điểm liên tục -> thứ tự hoàn hảo
    assert m["accuracy"] == 0.8
    assert sum(map(sum, m["confusion"])) == 5 and m["confusion"][1][2] == 1


def test_ndcg_uses_exponential_gain():
    # gain mũ: đặt nhãn 2 ở đầu đáng hơn hai nhãn 1 nhiều hơn gain tuyến tính
    from sklearn.metrics import ndcg_score
    y, s = [2, 1, 1, 0, 0], [10, 50, 40, 5, 1]
    lin = ndcg_score([y], [s], k=5)
    assert compute_ndcg_at_10(["J"] * 5, y, s) < lin


def test_spearman_per_jd_skips_constant_labels():
    from src.metrics import compute_spearman_per_jd
    jobs = ["A"] * 5 + ["B"] * 5
    y = [0, 1, 2, 0, 1] + [0] * 5          # JD B nhãn hằng -> bỏ
    s = [1, 2, 3, 1, 2] + [5, 4, 3, 2, 1]
    assert abs(compute_spearman_per_jd(jobs, y, s) - 1.0) < 1e-9


def test_paired_bootstrap_identical_methods_gives_zero():
    import pandas as pd
    from src.metrics import paired_bootstrap_ci
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"label": rng.integers(0, 3, 60), "pred": rng.integers(0, 3, 60),
                       "score": rng.random(60), "job_id": np.repeat([f"J{i}" for i in range(10)], 6)})
    f1 = lambda d: compute_macro_f1(d["label"], d["pred"])
    nd = lambda d: compute_ndcg_at_10(d["job_id"], d["label"], d["score"])
    assert paired_bootstrap_ci([(df, df)], f1, n_boot=50) == (0.0, 0.0, 0.0)
    assert paired_bootstrap_ci([(df, df)], nd, n_boot=20, by_jd=True) == (0.0, 0.0, 0.0)
    better = df.assign(pred=df["label"])
    point, lo, hi = paired_bootstrap_ci([(better, df)], f1, n_boot=200)
    assert point > 0 and lo > 0 and lo <= point <= hi

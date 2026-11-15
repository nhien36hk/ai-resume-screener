import numpy as np

from src.tuning import generate_weight_grid, search_best_thresholds, search_best_weights


def test_weight_grid_constraints():
    grid = generate_weight_grid()
    assert len(grid) == 3246, f"Lưới trọng số phải có đúng 3246 tổ hợp, thực tế: {len(grid)}"
    assert grid.shape[1] == 5

    # Tổng mỗi dòng xấp xỉ 1.0
    sums = np.sum(grid, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    # Mọi trọng số đều nằm trong [0.05, 0.50]
    assert np.all(grid >= 0.049)
    assert np.all(grid <= 0.501)


def test_search_best_weights_and_thresholds():
    # Giả lập dữ liệu 10 mẫu với 5 tiêu chí
    np.random.seed(42)
    scores = np.random.uniform(30, 90, size=(10, 5))
    mask = np.ones((10, 5), dtype=np.float32)
    y_true = np.array([0, 0, 1, 1, 1, 2, 2, 2, 1, 0])

    best_w, best_qwk = search_best_weights(scores, mask, y_true, t1=40, t2=70)
    assert len(best_w) == 5
    assert -1.0 <= best_qwk <= 1.0

    cont_scores = np.mean(scores, axis=1)
    best_t1, best_t2, best_f1 = search_best_thresholds(cont_scores, y_true, min_t1=30, max_t2=80, step=5)
    assert best_t1 < best_t2
    assert 0.0 <= best_f1 <= 1.0



def test_vectorized_combined_matches_scalar_formulas():
    """weighted_sum_scores + apply_must_have_penalty phải ra đúng như weighted_score + apply_must_have."""
    from src.scoring import apply_must_have, weighted_score
    from src.tuning import CRITERIA, apply_must_have_penalty, to_labels, weighted_sum_scores

    X = np.array([[80.0, 90.0, 100.0, 0.0, 0.0], [60.0, 50.0, 0.0, 70.0, 100.0]])
    M = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 1, 1]], dtype=float)  # 0 = tiêu chí bị skip
    w = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    failed = np.array([True, False])
    s = apply_must_have_penalty(weighted_sum_scores(X, M, w), failed, t2=70)

    for i in range(2):
        scores = {c: (X[i, k] if M[i, k] else None) for k, c in enumerate(CRITERIA)}
        expected = weighted_score(scores, dict(zip(CRITERIA, w)))
        expected, _ = apply_must_have(expected, int(failed[i]), 5.0, 0.0, t2=70)
        assert abs(s[i] - expected) < 1e-6
    assert s[0] == 69.0  # 92.86 bị chặn ở t2 - 1
    assert to_labels(np.array([39.9, 40.0, 69.9, 70.0]), 40, 70).tolist() == [0, 1, 1, 2]


def test_prepare_criteria_inputs():
    import pandas as pd
    from src.tuning import prepare_criteria_inputs

    # Kiểm tra với tên cột mới chuẩn
    df_new = pd.DataFrame([{
        "sim_chunk_200": 0.8,
        "score_skill_0.7": 85.0,
        "score_experience": 90.0,
        "score_education": 100.0,
        "score_certificate": np.nan,
        "miss_count_0.7": 0,
        "total_must_skills": 5,
        "has_insufficient_exp": False,
    }])
    scores, mask, skill_fail, exp_fail = prepare_criteria_inputs(df_new, skill_threshold=0.7, percentile_p5=0.2, percentile_p95=0.8)
    assert scores.shape == (1, 5)
    assert mask.shape == (1, 5)
    assert mask[0, 4] == 0.0
    assert not skill_fail[0]
    assert not exp_fail[0]

    # Kiểm tra các ngưỡng must_ratio
    df_must = pd.DataFrame([
        {"sim_chunk_200": 0.5, "score_skill_0.7": 80.0, "score_experience": 80.0, "score_education": 80.0, "score_certificate": 80.0,
         "miss_count_0.7": 1, "total_must_skills": 4, "has_insufficient_exp": False},  # cov = 3/4 = 0.75
    ])
    _, _, fail_0, _ = prepare_criteria_inputs(df_must, 0.7, 0.2, 0.8, must_ratio=0.0)
    assert not fail_0[0]

    _, _, fail_half, _ = prepare_criteria_inputs(df_must, 0.7, 0.2, 0.8, must_ratio=0.5)
    assert not fail_half[0]  # 0.75 >= 0.5 -> Đạt

    _, _, fail_strict, _ = prepare_criteria_inputs(df_must, 0.7, 0.2, 0.8, must_ratio=0.8)
    assert fail_strict[0]  # 0.75 < 0.8 -> Bị phạt

    _, _, fail_1, _ = prepare_criteria_inputs(df_must, 0.7, 0.2, 0.8, must_ratio=1.0)
    assert fail_1[0]  # miss=1 -> Bị phạt

    # Kiểm tra tương thích ngược với tên cột cũ
    df_old = pd.DataFrame([{
        "sim_chunk_200": 0.8, "skill_0.7": 85.0, "score_experience": 90.0, "score_education": 100.0,
        "score_certificate": 80.0, "must_miss_0.7": 0, "must_total": 3, "has_insufficient_exp": False,
    }])
    _, _, old_fail, _ = prepare_criteria_inputs(df_old, 0.7, 0.2, 0.8)
    assert not old_fail[0]

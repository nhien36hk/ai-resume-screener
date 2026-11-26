from experiments.e1_keyword import score_cv_keywords


def test_keyword_score_unweighted_exact_no_synonyms():
    cv = "Built services in Python and Docker on AWS."
    # 3/4 kỹ năng khớp nguyên văn, must và nice cùng trọng số, "aws" trùng lặp chỉ tính một lần
    assert score_cv_keywords(["python", "aws"], ["docker", "kubernetes", "aws"], cv) == 75.0
    # không dùng bảng đồng nghĩa: "js" không khớp "javascript"
    assert score_cv_keywords(["javascript"], [], "I write js daily") == 0.0
    assert score_cv_keywords([], [], cv) == 0.0

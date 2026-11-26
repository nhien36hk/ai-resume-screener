from src.scoring import (
    apply_must_have,
    compute_cert_score,
    compute_edu_score,
    compute_exp_score,
    weighted_score,
)


def test_experience_scoring():
    assert compute_exp_score(cv_years_exp=2.0, jd_min_years_exp=4.0) == 50.0
    assert compute_exp_score(cv_years_exp=5.0, jd_min_years_exp=4.0) == 100.0
    assert compute_exp_score(cv_years_exp=3.0, jd_min_years_exp=0.0) is None  # Skip


def test_education_scoring():
    # cv master >= jd bachelor -> degree 100, same major -> 100
    assert compute_edu_score("master", "bachelor", "computer_science", "computer_science") == 100.0
    # cv bachelor vs jd master -> degree 50 (thiếu 1 bậc), same major -> 0.7*50 + 0.3*100 = 65.0
    assert compute_edu_score("bachelor", "master", "computer_science", "computer_science") == 65.0
    # cv high_school vs jd master -> degree 0 (thiếu 2 bậc)
    assert compute_edu_score("high_school", "master", "computer_science", "computer_science") == 30.0
    # JD không yêu cầu cả 2 -> Skip
    assert compute_edu_score("bachelor", "none", "unknown", "unknown") is None
    # major other == other không tính là khớp (D1)
    assert compute_edu_score("bachelor", "bachelor", "other", "other") == 70.0


def test_cert_scoring():
    assert compute_cert_score(["AWS Certified", "CPA"], ["AWS", "PMP"]) == 50.0
    assert compute_cert_score(["CPA"], []) is None  # Skip


def test_must_have_penalty():
    # Ứng viên 90 điểm nhưng thiếu 1 must skill -> bị ép xuống t2 - 1 = 69
    s_penalized, reasons = apply_must_have(
        score=90.0, missing_must_skills=1, cv_years_exp=5.0, jd_min_years_exp=3.0, t2=70.0
    )
    assert s_penalized == 69.0
    assert "missing_1_must_skills" in reasons

    # Ứng viên kinh nghiệm < 0.5 * JD -> bị ép xuống 69
    s_penalized, reasons = apply_must_have(
        score=85.0, missing_must_skills=0, cv_years_exp=1.0, jd_min_years_exp=4.0, t2=70.0
    )
    assert s_penalized == 69.0
    assert "exp_below_half" in reasons

    # Không vi phạm -> Giữ nguyên điểm
    s_ok, reasons = apply_must_have(
        score=85.0, missing_must_skills=0, cv_years_exp=4.0, jd_min_years_exp=4.0, t2=70.0
    )
    assert s_ok == 85.0
    assert len(reasons) == 0


def test_dynamic_weight_aggregation():
    scores = {"sem": 80.0, "exp": 100.0, "cert": None}  # cert bị skip
    weights = {"sem": 0.5, "exp": 0.3, "cert": 0.2}
    # Mẫu số chỉ còn 0.5 + 0.3 = 0.8
    # Tử số: 0.5 * 80 + 0.3 * 100 = 40 + 30 = 70
    # Điểm: 70 / 0.8 = 87.5
    s = weighted_score(scores, weights)
    assert abs(s - 87.5) < 1e-4



def test_education_degree_not_required_keeps_spec_formula():
    # JD không yêu cầu bậc (none) nhưng yêu cầu ngành: S_degree = 100 -> 0.7*100 + 0.3*S_major
    assert compute_edu_score("high_school", "none", "marketing_communications", "accounting_finance") == 70.0
    assert compute_edu_score("high_school", "none", "accounting_finance", "accounting_finance") == 100.0


def test_apply_must_have_ratio_rule():
    # 1/4 must thiếu: tỉ lệ khớp 0.75 -> đạt với ρ=0.75, vi phạm với ρ=1 (luật gốc)
    ok, reasons = apply_must_have(80.0, 1, 5.0, 0.0, t2=70, must_total=4, must_ratio=0.75)
    assert ok == 80.0 and reasons == []
    capped, reasons = apply_must_have(80.0, 1, 5.0, 0.0, t2=70, must_total=4, must_ratio=1.0)
    assert capped == 69.0 and reasons

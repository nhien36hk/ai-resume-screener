# Level 1: Thư viện chuẩn Python (Standard Library)
from typing import Dict, List, Optional, Tuple

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import numpy as np

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


DEFAULT_WEIGHTS = {
    "sem": 0.30,
    "skill": 0.30,
    "exp": 0.20,
    "edu": 0.10,
    "cert": 0.10,
}

DEFAULT_T2_THRESHOLD = 70.0


class MatchScorer:
    """Tổng hợp điểm động có trọng số và áp dụng chế tài điểm liệt Must-Have theo Mục 4 plan.md."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        t2_threshold: float = DEFAULT_T2_THRESHOLD,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.t2_threshold = t2_threshold

    def aggregate_score(self, criteria_scores: Dict[str, Optional[float]]) -> float:
        """
        Tính điểm gộp có trọng số động:
        S = ∑_{i ∈ A} (w_i * S_i) / ∑_{i ∈ A} w_i
        (Các tiêu chí bị None sẽ bị loại khỏi mẫu số).
        """
        weighted_sum = 0.0
        total_w = 0.0

        for key, w in self.weights.items():
            s = criteria_scores.get(key)
            if s is not None:
                weighted_sum += w * s
                total_w += w

        if total_w <= 0.0:
            return 0.0

        return float(weighted_sum / total_w)

    def apply_must_have_penalty(
        self,
        raw_score: float,
        missing_mandatory_skill: bool,
        e_jd: float,
        e_cv: float,
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Must-have Penalty:
        ĐIỀU KIỆN: Thiếu >= 1 kỹ năng bắt buộc HOẶC E_CV < 0.5 * E_JD.
        HÀNH ĐỘNG: S_final = min(S, t2 - 1).
        """
        violated = False
        reasons = []

        if missing_mandatory_skill:
            violated = True
            reasons.append("Thiếu kỹ năng bắt buộc")

        if e_jd > 0 and e_cv < 0.5 * e_jd:
            violated = True
            reasons.append(f"Kinh nghiệm không đủ 50% ({e_cv:.1f} năm < 0.5 * {e_jd:.1f} năm)")

        if violated:
            penalty_score = min(raw_score, self.t2_threshold - 1.0)
            return float(penalty_score), True, "; ".join(reasons)

        return float(raw_score), False, None

    def score_single(
        self,
        criteria_scores: Dict[str, Optional[float]],
        missing_mandatory_skill: bool = False,
        e_jd: float = 0.0,
        e_cv: float = 0.0,
    ) -> Dict[str, any]:
        """Quy trình tính điểm hoàn chỉnh cho 1 cặp CV - JD."""
        raw_score = self.aggregate_score(criteria_scores)
        final_score, failed, reason = self.apply_must_have_penalty(
            raw_score=raw_score,
            missing_mandatory_skill=missing_mandatory_skill,
            e_jd=e_jd,
            e_cv=e_cv,
        )

        return {
            "raw_score": raw_score,
            "final_score": final_score,
            "must_have_failed": failed,
            "must_have_reason": reason,
        }

# Level 1: Thư viện chuẩn Python (Standard Library)
import re
from typing import Dict, List, Optional, Set, Tuple

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import numpy as np

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


DEGREE_LEVELS = {
    "phd": 4,
    "doctorate": 4,
    "master": 3,
    "ms": 3,
    "mba": 3,
    "bachelor": 2,
    "bs": 2,
    "ba": 2,
    "undergraduate": 2,
    "associate": 1,
    "high school": 0,
}

MAJORS_KEYWORDS = [
    "computer science", "software", "information technology", "engineering",
    "accounting", "finance", "economics", "business", "mathematics", "statistics",
    "data science", "cybersecurity",
]

COMMON_CERTS = [
    "aws", "azure", "gcp", "pmp", "cpa", "cfa", "cissp", "scrum", "agile",
    "cisa", "comptia", "ccna", "ccnp", "itil", "six sigma", "prince2", "series 7",
]


class CriteriaExtractor:
    """Bóc tách và tính toán 4 tiêu chí rời rạc (Kỹ năng, Kinh nghiệm, Học vấn, Chứng chỉ)."""

    def __init__(self, tau_skill: float = 0.70):
        self.tau_skill = tau_skill

    # ---------------- 1. KỸ NĂNG (S_skill) ----------------
    def extract_jd_skills(self, jd_text: str) -> List[Tuple[str, int]]:
        """
        Bóc tách danh sách yêu cầu kỹ năng từ JD kèm trọng số:
        Bắt buộc (required, must have): w = 2.
        Ưu tiên (preferred, plus, nice to have): w = 1.
        """
        lines = re.split(r"[\n\r]+|[•\-\*]\s+", jd_text)
        skill_items = []
        is_preferred_section = False

        for raw_line in lines:
            line = raw_line.strip()
            if len(line) < 5:
                continue

            lower_line = line.lower()
            if any(p in lower_line for p in ["preferred", "plus", "nice to have", "bonus"]):
                is_preferred_section = True
            elif any(r in lower_line for r in ["required", "qualification", "must have", "what we're looking for"]):
                is_preferred_section = False

            # Nếu là dòng mô tả yêu cầu công việc
            if any(k in lower_line for k in ["experience", "knowledge", "proficien", "skill", "technolog", "ability", "hands-on"]):
                weight = 1 if is_preferred_section else 2
                skill_items.append((line, weight))

        return skill_items

    def calculate_skill_score(
        self,
        jd_skills: List[Tuple[str, int]],
        cv_text: str,
        cv_chunks_embs: Optional[np.ndarray] = None,
        skill_embs: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[float], bool]:
        """
        Tính S_skill = 100 * ∑(w_s * m(s, c)) / ∑ w_s.
        Trả về: (điểm S_skill, cờ thiếu_kỹ_năng_bắt_buộc).
        """
        if not jd_skills:
            return None, False

        cv_lower = cv_text.lower()
        total_w = sum(w for _, w in jd_skills)
        matched_w = 0.0
        missing_mandatory = False

        for idx, (skill_text, weight) in enumerate(jd_skills):
            skill_lower = skill_text.lower()
            # 1. Khớp exact/từ khóa quan trọng
            words = [w for w in re.findall(r"\b[a-zA-Z0-9\+#\.]+\b", skill_lower) if len(w) > 2]
            exact_match = any(w in cv_lower for w in words[:3]) if words else False

            m_val = 0.0
            if exact_match:
                m_val = 1.0
            elif cv_chunks_embs is not None and skill_embs is not None:
                # 2. Khớp cosine semantic similarity
                cos_sims = np.dot(cv_chunks_embs, skill_embs[idx])
                max_cos = float(np.max(cos_sims)) if len(cos_sims) > 0 else 0.0
                if max_cos > self.tau_skill:
                    m_val = max_cos

            matched_w += weight * m_val
            if weight == 2 and m_val < 0.5:
                missing_mandatory = True

        score = (matched_w / total_w) * 100.0 if total_w > 0 else 100.0
        return float(np.clip(score, 0.0, 100.0)), missing_mandatory

    # ---------------- 2. KINH NGHIỆM (S_exp) ----------------
    @staticmethod
    def extract_years_required(text: str) -> Optional[float]:
        """Trích xuất số năm kinh nghiệm yêu cầu trong JD."""
        matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|experience|work)?", text, re.IGNORECASE)
        if matches:
            years = [float(m) for m in matches if 0 < float(m) <= 25]
            if years:
                return max(years)
        return None

    @staticmethod
    def extract_years_in_cv(text: str) -> float:
        """Trích xuất tổng số năm kinh nghiệm từ các mốc thời gian trong CV."""
        # Bắt các cặp năm: 2018 - 2023 hoặc 2019 - Present
        year_pairs = re.findall(r"\b(20\d\d|19\d\d)\s*[-–—to]+\s*(20\d\d|present|current)\b", text, re.IGNORECASE)
        intervals = []
        for start, end in year_pairs:
            s = int(start)
            e = 2026 if end.lower() in ["present", "current"] else int(end)
            if 1980 <= s <= e <= 2030:
                intervals.append((s, e))

        if intervals:
            # Hợp nhất các khoảng giao nhau (merge overlapping intervals)
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            for current in intervals[1:]:
                prev_s, prev_e = merged[-1]
                if current[0] <= prev_e:
                    merged[-1] = (prev_s, max(prev_e, current[1]))
                else:
                    merged.append(current)
            total_years = sum(e - s for s, e in merged)
            return float(max(total_years, 0.5))

        # Fallback: Quét các câu đề cập "X years of experience"
        direct = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|experience|work)", text, re.IGNORECASE)
        if direct:
            return float(max(float(d) for d in direct))

        return 1.0

    def calculate_experience_score(self, jd_text: str, cv_text: str) -> Tuple[Optional[float], float, float]:
        """Tính S_exp = 100 * min(E_CV / E_JD, 1). Skip nếu JD không yêu cầu."""
        e_jd = self.extract_years_required(jd_text)
        if e_jd is None or e_jd <= 0:
            return None, 0.0, 0.0

        e_cv = self.extract_years_in_cv(cv_text)
        s_exp = 100.0 * min(e_cv / e_jd, 1.0)
        return float(s_exp), e_jd, e_cv

    # ---------------- 3. HỌC VẤN (S_edu) ----------------
    @staticmethod
    def extract_degree_level(text: str) -> Optional[int]:
        """Xác định bậc học cao nhất: 4: PhD, 3: Master, 2: Bachelor, 1: Associate, 0: High School."""
        text_lower = text.lower()
        max_level = None
        for deg, level in DEGREE_LEVELS.items():
            pattern = rf"\b{re.escape(deg)}\b"
            if re.search(pattern, text_lower):
                if max_level is None or level > max_level:
                    max_level = level
        return max_level

    @staticmethod
    def extract_major_match(jd_text: str, cv_text: str) -> float:
        """Kiểm tra khớp chuyên ngành đào tạo."""
        jd_lower = jd_text.lower()
        cv_lower = cv_text.lower()
        for major in MAJORS_KEYWORDS:
            if major in jd_lower and major in cv_lower:
                return 100.0
        return 50.0

    def calculate_education_score(self, jd_text: str, cv_text: str) -> Optional[float]:
        """S_edu = 0.7 * S_degree + 0.3 * S_major."""
        jd_deg = self.extract_degree_level(jd_text)
        cv_deg = self.extract_degree_level(cv_text)

        if jd_deg is None:
            return None

        cv_deg_val = cv_deg if cv_deg is not None else 1
        if cv_deg_val >= jd_deg:
            s_degree = 100.0
        elif cv_deg_val == jd_deg - 1:
            s_degree = 50.0
        else:
            s_degree = 0.0

        s_major = self.extract_major_match(jd_text, cv_text)
        s_edu = 0.7 * s_degree + 0.3 * s_major
        return float(s_edu)

    # ---------------- 4. CHỨNG CHỈ (S_cert) ----------------
    @staticmethod
    def calculate_certificate_score(jd_text: str, cv_text: str) -> Optional[float]:
        """S_cert = 100 * (matched_cert / required_cert). Skip nếu JD không yêu cầu."""
        jd_lower = jd_text.lower()
        cv_lower = cv_text.lower()

        required = [c for c in COMMON_CERTS if re.search(rf"\b{re.escape(c)}\b", jd_lower)]
        if not required:
            return None

        matched = sum(1 for c in required if re.search(rf"\b{re.escape(c)}\b", cv_lower))
        score = 100.0 * (matched / len(required))
        return float(score)

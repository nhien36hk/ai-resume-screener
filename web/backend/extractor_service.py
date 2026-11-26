"""Trích xuất thực thể có cấu trúc cho JD và CV (ưu tiên cache MD5, gọi NVIDIA API nếu có khóa, hoặc fallback cục bộ)."""
import os
import re
from typing import Any, Dict, List, Tuple

from src.config import CV_CACHE_PATH, CV_HEADERS, DEGREE_LEVELS, JD_CACHE_PATH
from src.data.extractor import NvidiaAPIBackend, extract_many
from src.features.skills import SKILL_SYNONYMS, normalize_skill
from src.features.text import prepare_finetune_text
from src.utils import compute_text_hash, load_json

LOCAL_SKILL_LEXICON = tuple(sorted({
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "php", "ruby", "r",
    "sql", "mysql", "postgresql", "mssql", "mongodb", "redis", "oracle", "sqlite",
    "html", "css", "react", "vue", "angular", "node.js", "django", "flask", "fastapi", "spring",
    "excel", "power bi", "tableau", "looker", "qlik", "sas", "spss", "vba", "erp", "sap",
    "machine learning", "deep learning", "natural language processing", "computer vision",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "scipy", "spark", "hadoop", "airflow",
    "docker", "kubernetes", "aws", "azure", "google cloud platform", "linux", "git", "cicd", "jenkins",
    "data analysis", "data modeling", "data visualization", "etl", "reporting", "statistics",
    "project management", "agile", "scrum", "jira", "salesforce", "accounting", "auditing", "financial reporting",
    "communication", "teamwork", "leadership", "problem solving", "english",
    *SKILL_SYNONYMS.keys(),
    *SKILL_SYNONYMS.values(),
}, key=len, reverse=True))

PII_LINE_RE = re.compile(
    r"^\s*(?:họ\s*(?:và\s*)?tên|name|full\s*name|ngày\s*sinh|date\s*of\s*birth|dob|sinh\s*ngày|"
    r"giới\s*tính|gender|sex|email|e-mail|sđt|điện\s*thoại|phone|mobile|tel|địa\s*chỉ|address|quê\s*quán)\s*:",
    re.IGNORECASE,
)

DEGREE_PATTERNS = [
    ("phd", re.compile(r"\b(ph\.?d|doctorate|doctoral|tiến\s*sĩ)\b", re.IGNORECASE)),
    ("master", re.compile(r"\b(master|m\.?sc|m\.?s\.?|mba|thạc\s*sĩ)\b", re.IGNORECASE)),
    ("bachelor", re.compile(r"\b(bachelor|b\.?sc|b\.?s\.?|b\.?a\.?|cử\s*nhân|kỹ\s*sư|đại\s*học)\b", re.IGNORECASE)),
    ("associate", re.compile(r"\b(associate|cao\s*đẳng)\b", re.IGNORECASE)),
    ("high_school", re.compile(r"\b(high\s*school|ged|trung\s*học\s*phổ\s*thông|thpt)\b", re.IGNORECASE)),
]

CERT_PATTERNS = (
    "aws certified", "pmp", "cpa", "cfa", "scrum master", "cka", "azure", "google data analytics",
    "ielts", "toeic", "cissp", "comptia", "oracle certified", "salesforce certified",
)

MAJOR_KEYWORDS = {
    "data_science_ai": ("data science", "artificial intelligence", "machine learning", "khoa học dữ liệu", "trí tuệ nhân tạo"),
    "computer_science": ("computer science", "khoa học máy tính"),
    "software_engineering": ("software engineering", "kỹ thuật phần mềm", "công nghệ phần mềm"),
    "information_technology": ("information technology", "information systems", "công nghệ thông tin", "hệ thống thông tin"),
    "accounting_finance": ("accounting", "finance", "kế toán", "tài chính"),
    "business_administration": ("business administration", "management", "quản trị kinh doanh"),
    "economics": ("economics", "kinh tế"),
    "mathematics_statistics": ("mathematics", "statistics", "toán", "thống kê"),
    "electrical_electronics": ("electrical", "electronics", "điện", "điện tử"),
    "mechanical_engineering": ("mechanical", "cơ khí"),
}

_JD_CACHE: Dict[str, Any] | None = None
_CV_CACHE: Dict[str, Any] | None = None


def _get_caches() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    global _JD_CACHE, _CV_CACHE
    if _JD_CACHE is None:
        _JD_CACHE = load_json(JD_CACHE_PATH, {})
    if _CV_CACHE is None:
        _CV_CACHE = load_json(CV_CACHE_PATH, {})
    return _JD_CACHE, _CV_CACHE


def strip_cv_pii_for_scoring(text: str) -> str:
    """Bỏ phần thông tin cá nhân đầu CV (nhất quán V7 + bảo đảm công bằng tên/ngày sinh)."""
    body = prepare_finetune_text(text)
    lines = [ln for ln in body.replace("\r", "").split("\n")]
    filtered: List[str] = []
    in_header = True
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if in_header and PII_LINE_RE.match(s):
            continue
        if in_header and len(s.split()) <= 5 and not any(ch.isdigit() for ch in s) and "@" not in s:
            # Nếu dòng đầu là tên riêng ngắn đứng ngay trước các dòng PII/mục nội dung
            if any(h.lower() in s.lower() for h in (*CV_HEADERS, "tóm tắt", "kỹ năng", "kinh nghiệm", "học vấn", "chứng chỉ")):
                in_header = False
                filtered.append(ln)
            elif len(filtered) == 0 and any(PII_LINE_RE.match(nxt.strip()) for nxt in lines[1:4] if nxt.strip()):
                continue
            else:
                in_header = False
                filtered.append(ln)
        else:
            in_header = False
            filtered.append(ln)
    return "\n".join(filtered).strip() or body


def split_sections(text: str) -> Dict[str, str]:
    headers = (*CV_HEADERS, "Tóm tắt", "Kỹ năng", "Kinh nghiệm", "Học vấn", "Chứng chỉ", "Dự án")
    pattern = re.compile(rf"^\s*({'|'.join(re.escape(h) for h in headers)})\b[:\s-]*$", re.IGNORECASE)
    sections: Dict[str, List[str]] = {"overview": []}
    current = "overview"
    for ln in text.replace("\r", "").split("\n"):
        s = ln.strip()
        if not s:
            continue
        m = pattern.match(s)
        if m:
            current = m.group(1).strip().lower()
            sections.setdefault(current, [])
        else:
            sections[current].append(s)
    return {k: "\n".join(v) for k, v in sections.items() if v}


def _find_skills(text: str) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    seen = set()
    for skill in LOCAL_SKILL_LEXICON:
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", lowered):
            norm = normalize_skill(skill)
            if norm not in seen:
                seen.add(norm)
                found.append(norm)
    return found


def _extract_years(text: str) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|năm)", text, re.IGNORECASE)
    if matches:
        return max(float(v) for v in matches)
    ranges = re.findall(r"(20\d{2}|19\d{2})\s*[-–to]+\s*(20\d{2}|present|now|hiện tại)", text, re.IGNORECASE)
    if ranges:
        total = 0.0
        for start_s, end_s in ranges:
            start = int(start_s)
            end = 2026 if not end_s.isdigit() else int(end_s)
            if 0 <= end - start <= 40:
                total += max(1.0, float(end - start))
        return min(total, 40.0)
    return 0.0


def _extract_degree(text: str) -> str:
    best_idx, best_deg = 0, "none"
    for deg, pat in DEGREE_PATTERNS:
        if pat.search(text):
            idx = DEGREE_LEVELS.index(deg)
            if idx > best_idx:
                best_idx, best_deg = idx, deg
    return best_deg


def _extract_major_group(text: str) -> str:
    lowered = text.lower()
    for group, kws in MAJOR_KEYWORDS.items():
        if any(kw in lowered for kw in kws):
            return group
    return "unknown"


def _extract_certs(text: str) -> List[str]:
    lowered = text.lower()
    return [c for c in CERT_PATTERNS if c in lowered]


def extract_jd_local(jd_text: str, title: str = "") -> Dict[str, Any]:
    lines = [ln.strip() for ln in jd_text.replace("\r", "").split("\n") if ln.strip()]
    must_lines, nice_lines = [], []
    mode = "must"
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in ("preferred", "nice to have", "bonus", "plus", "ưu tiên")):
            mode = "nice"
        elif any(k in low for k in ("required", "must have", "mandatory", "minimum", "bắt buộc", "yêu cầu")):
            mode = "must"
        (nice_lines if mode == "nice" else must_lines).append(ln)

    must_skills = _find_skills("\n".join(must_lines))[:8]
    nice_skills = [s for s in _find_skills("\n".join(nice_lines)) if s not in set(must_skills)][:10]
    if not must_skills and nice_skills:
        must_skills, nice_skills = nice_skills[:5], nice_skills[5:]

    return {
        "job_title": title.strip() or (lines[0][:80] if lines else "Vị trí tuyển dụng"),
        "must_have_skills": must_skills,
        "nice_to_have_skills": nice_skills,
        "min_years_experience": _extract_years(jd_text),
        "required_certs": _extract_certs(jd_text),
        "education_evidence_ids": [],
        "min_degree": _extract_degree(jd_text),
        "required_majors": [],
        "major_group": _extract_major_group(jd_text),
        "accepts_related_field": bool(re.search(r"related\s+field|equivalent|ngành\s+liên\s+quan", jd_text, re.IGNORECASE)),
    }


def extract_cv_local(cv_text: str) -> Dict[str, Any]:
    clean_body = strip_cv_pii_for_scoring(cv_text)
    return {
        "candidate_title": "",
        "skills": _find_skills(clean_body),
        "total_years_experience": _extract_years(clean_body),
        "certs": _extract_certs(clean_body),
        "education_evidence_ids": [],
        "highest_degree": _extract_degree(clean_body),
        "majors": [],
        "major_group": _extract_major_group(clean_body),
    }


def extract_entities(kind: str, text: str, title: str = "") -> Tuple[Dict[str, Any], str]:
    """Trả về (entities_dict, source) cho JD hoặc CV."""
    jd_cache, cv_cache = _get_caches()
    key = compute_text_hash(text)
    cache = jd_cache if kind == "jd" else cv_cache
    if key in cache and isinstance(cache[key], dict):
        return dict(cache[key]), "cache"

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("API_NVIDIA")
    if api_key and os.getenv("WEB_DISABLE_API", "0") != "1":
        try:
            backend = NvidiaAPIBackend()
            target_text = text if kind == "jd" else strip_cv_pii_for_scoring(text)
            res = extract_many(kind, [target_text], backend)
            if res and isinstance(res[0], dict):
                return res[0], "nvidia_api"
        except Exception:
            pass

    if kind == "jd":
        return extract_jd_local(text, title=title), "local_fallback"
    return extract_cv_local(text), "local_fallback"

import re
from typing import Dict, List, Optional, Set

import numpy as np

SKILL_SYNONYMS: Dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "vuejs": "vue",
    "vue.js": "vue",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "psql": "postgresql",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "qa": "quality assurance",
    "qc": "quality control",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "c#": "c sharp",
    "cpp": "c++",
    "go": "golang",  # chuẩn hoá về "golang": dạng "go" khớp nhầm chữ "go" thường trong text CV
    "sql server": "mssql",
    "ms sql": "mssql",
}


def normalize_skill(skill: str, synonyms: Optional[Dict[str, str]] = None) -> str:
    cleaned = re.sub(r"[^\w\s\+\#\.\/]", " ", skill.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    lookup = synonyms if synonyms is not None else SKILL_SYNONYMS
    return lookup.get(cleaned, cleaned)


def is_exact_match(jd_skill: str, cv_text: str, synonyms: Optional[Dict[str, str]] = None) -> bool:
    """Kỹ năng JD (dạng gốc hoặc dạng chuẩn hoá) xuất hiện nguyên văn trong text CV."""
    text = cv_text.lower()
    forms = {jd_skill.strip().lower(), normalize_skill(jd_skill, synonyms)}
    # lookaround thay cho \b: \b không có biên sau "+" / "#", nên "c++", "c#" không bao giờ khớp
    return any(re.search(rf"(?<!\w){re.escape(f)}(?!\w)", text) for f in forms if f)


def match_single_skill(
    jd_skill_norm: str,
    jd_skill_embedding: Optional[np.ndarray],
    cv_skills_norm: Set[str],
    cv_skill_embeddings: Optional[np.ndarray],
    skill_threshold: float = 0.7,
) -> float:
    """Trả về mức độ khớp kỹ năng: 1.0 nếu trùng tên/đồng nghĩa; cosine nếu > skill_threshold; ngược lại 0.0."""
    if jd_skill_norm in cv_skills_norm:
        return 1.0

    if jd_skill_embedding is not None and cv_skill_embeddings is not None and len(cv_skill_embeddings) > 0:
        sims = np.dot(cv_skill_embeddings, jd_skill_embedding)
        max_sim = float(np.max(sims))
        if max_sim > skill_threshold:
            return max_sim

    return 0.0

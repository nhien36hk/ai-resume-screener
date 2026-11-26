from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.config import DEGREE_LEVELS
from src.features.skills import match_single_skill, normalize_skill

DEGREE_ORDER = {deg: idx for idx, deg in enumerate(DEGREE_LEVELS)}


# ── Mục 2: Ngữ nghĩa ─────────────────────────────────────────────────────────

def max_mean_sim(jd_chunk_embs: np.ndarray, cv_chunk_embs: np.ndarray) -> float:
    """sim = mean_r max_p (Q · P^T) tích vô hướng giữa chunk JD và chunk CV."""
    if jd_chunk_embs is None or cv_chunk_embs is None:
        return 0.0
    if len(jd_chunk_embs) == 0 or len(cv_chunk_embs) == 0:
        return 0.0
    sim_matrix = np.dot(jd_chunk_embs, cv_chunk_embs.T)
    max_per_jd = np.max(sim_matrix, axis=1)
    return float(np.mean(max_per_jd))


def scale_p5_p95(sim: float, percentile_p5: float, percentile_p95: float) -> float:
    """Chuẩn hóa điểm tương đồng sim về thang điểm 100 dựa trên phân vị p5 và p95."""
    denom = percentile_p95 - percentile_p5
    if denom <= 0:
        return 50.0
    normalized = (sim - percentile_p5) / denom
    return 100.0 * float(np.clip(normalized, 0.0, 1.0))


# ── Mục 3: 4 Tiêu chí rời rạc ────────────────────────────────────────────────

def compute_skill_score(
    must_skills: List[str],
    nice_skills: List[str],
    cv_skills: List[str],
    must_skill_embeddings: Optional[np.ndarray] = None,
    nice_skill_embeddings: Optional[np.ndarray] = None,
    cv_skill_embeddings: Optional[np.ndarray] = None,
    skill_threshold: float = 0.7,
    synonyms: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], int]:
    """Score_skill = 100 * sum(w * m) / sum(w) (must w=2, nice w=1). Trả về (score, missing_must)."""
    if not must_skills and not nice_skills:
        return None, 0

    cv_skills_norm = {normalize_skill(s, synonyms) for s in cv_skills}
    total_weight = len(must_skills) * 2.0 + len(nice_skills) * 1.0
    if total_weight == 0:
        return None, 0

    weighted_match_sum = 0.0
    missing_must_count = 0

    for i, skill in enumerate(must_skills):
        jd_skill_norm = normalize_skill(skill, synonyms)
        jd_skill_embedding = must_skill_embeddings[i] if must_skill_embeddings is not None and i < len(must_skill_embeddings) else None
        skill_match_score = match_single_skill(
            jd_skill_norm, jd_skill_embedding, cv_skills_norm, cv_skill_embeddings, skill_threshold
        )
        weighted_match_sum += 2.0 * skill_match_score
        if skill_match_score == 0.0:
            missing_must_count += 1

    for j, skill in enumerate(nice_skills):
        jd_skill_norm = normalize_skill(skill, synonyms)
        jd_skill_embedding = nice_skill_embeddings[j] if nice_skill_embeddings is not None and j < len(nice_skill_embeddings) else None
        skill_match_score = match_single_skill(
            jd_skill_norm, jd_skill_embedding, cv_skills_norm, cv_skill_embeddings, skill_threshold
        )
        weighted_match_sum += 1.0 * skill_match_score

    score = 100.0 * (weighted_match_sum / total_weight)
    return float(np.clip(score, 0.0, 100.0)), missing_must_count


def compute_exp_score(cv_years_exp: float, jd_min_years_exp: float) -> Optional[float]:
    """Score_exp = 100 * min(E_CV / E_JD, 1). Skip nếu E_JD <= 0."""
    if jd_min_years_exp is None or jd_min_years_exp <= 0.0:
        return None
    ratio = max(0.0, float(cv_years_exp)) / float(jd_min_years_exp)
    return 100.0 * min(ratio, 1.0)


def compute_edu_score(
    cv_degree: str,
    jd_degree: str,
    cv_major_group: str = "unknown",
    jd_major_group: str = "unknown",
) -> Optional[float]:
    """S_edu = 0.7 * S_degree + 0.3 * S_major."""
    has_degree_req = jd_degree in DEGREE_ORDER and DEGREE_ORDER[jd_degree] > 0
    has_major_req = jd_major_group not in {"unknown", "none"}

    if not has_degree_req and not has_major_req:
        return None

    cv_deg_idx = DEGREE_ORDER.get(cv_degree, 0)
    jd_deg_idx = DEGREE_ORDER.get(jd_degree, 0)

    if cv_deg_idx >= jd_deg_idx:
        score_degree = 100.0
    elif cv_deg_idx == jd_deg_idx - 1:
        score_degree = 50.0
    else:
        score_degree = 0.0

    if not has_major_req:
        return score_degree

    # Điểm chuyên ngành = 100 nếu trùng major_group
    is_major_match = (
        cv_major_group == jd_major_group and cv_major_group not in {"other", "unknown"}
    )
    score_major = 100.0 if is_major_match else 0.0
    return 0.7 * score_degree + 0.3 * score_major


def compute_cert_score(cv_certs: List[str], jd_certs: List[str]) -> Optional[float]:
    """S_cert = 100 * (matched / required). Skip nếu JD không yêu cầu."""
    if not jd_certs:
        return None

    norm_cv_certs = [c.strip().lower() for c in cv_certs if c.strip()]
    matched = 0
    for req in jd_certs:
        req_norm = req.strip().lower()
        if any(req_norm in c or c in req_norm for c in norm_cv_certs):
            matched += 1

    return 100.0 * min(matched / len(jd_certs), 1.0)


# ── Mục 4: Tổng hợp điểm & Điểm liệt Must-have ──────────────────────────────

def weighted_score(
    scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> float:
    """S = sum_{i in A} (w_i * S_i) / sum_{i in A} w_i."""
    weighted_sum = 0.0
    weight_total = 0.0

    for key, score in scores.items():
        if score is not None and key in weights:
            w = weights[key]
            weighted_sum += w * score
            weight_total += w

    if weight_total <= 0.0:
        return 0.0
    return weighted_sum / weight_total


def apply_must_have(
    score: float,
    missing_must_skills: int,
    cv_years_exp: float,
    jd_min_years_exp: float,
    t2: float = 70.0,
    must_total: int = 0,
    must_ratio: float = 1.0,
) -> Tuple[float, List[str]]:
    """Nếu tỉ lệ must-skill khớp < must_ratio HOẶC E_CV < 0.5 * E_JD: S_final = min(S, t2 - 1).
    must_total=0 (không biết tổng): luật gốc, thiếu >= 1 must-skill là vi phạm (giống must_ratio=1)."""
    failed_reasons = []

    skill_fail = missing_must_skills >= 1 if must_total <= 0 else 1.0 - missing_must_skills / must_total < must_ratio
    if skill_fail:
        failed_reasons.append(f"missing_{missing_must_skills}_must_skills")

    if jd_min_years_exp is not None and jd_min_years_exp > 0.0:
        if cv_years_exp < 0.5 * jd_min_years_exp:
            failed_reasons.append("exp_below_half")

    if failed_reasons:
        final_score = min(score, float(t2 - 1.0))
        return max(0.0, final_score), failed_reasons

    return max(0.0, min(100.0, score)), []


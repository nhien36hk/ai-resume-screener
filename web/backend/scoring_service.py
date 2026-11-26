"""Dịch vụ nạp mô hình SentenceTransformer và chấm điểm ứng viên theo đúng pipeline src/."""
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_SEQ_LENGTH
from src.features.skills import SKILL_SYNONYMS, match_single_skill, normalize_skill
from src.features.text import chunk_text
from src.scoring import (
    apply_must_have,
    compute_cert_score,
    compute_edu_score,
    compute_exp_score,
    compute_skill_score,
    max_mean_sim,
    scale_p5_p95,
    weighted_score,
)
from src.utils import load_json
from web.backend.config import ENCODE_BATCH_SIZE, LEVEL_LABELS, MODEL_CONFIG_PATH, MODEL_SOURCE, MODEL_VERSION
from web.backend.extractor_service import strip_cv_pii_for_scoring


class ScoringEngine:
    """Nạp một lần mô hình và cấu hình tham số tối ưu trên tập kiểm định."""

    def __init__(self) -> None:
        cfg_data = load_json(MODEL_CONFIG_PATH, {})
        if not cfg_data or "config" not in cfg_data:
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình mô hình: {MODEL_CONFIG_PATH}")
        self.cfg = cfg_data["config"]
        self.model_version = MODEL_VERSION
        self.model = SentenceTransformer(MODEL_SOURCE)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        self.tokenizer = self.model.tokenizer

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def build_jd_cache(self, jd_text: str, must_skills: List[str], nice_skills: List[str]) -> Dict[str, Any]:
        chunks = chunk_text(jd_text, self.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
        chunk_embs = self.encode_texts(chunks)
        must_embs = self.encode_texts(must_skills) if must_skills else np.zeros((0, 384), dtype=np.float32)
        nice_embs = self.encode_texts(nice_skills) if nice_skills else np.zeros((0, 384), dtype=np.float32)
        return {
            "chunks": chunks,
            "chunk_embs": chunk_embs.tolist(),
            "must_embs": must_embs.tolist(),
            "nice_embs": nice_embs.tolist(),
        }

    def evaluate_candidates_batch(
        self,
        jd_extracted: Dict[str, Any],
        jd_cache: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Chấm lô toàn bộ CV của một JD và trả về (danh sách kết quả, thời gian từng bước ms)."""
        t0 = time.perf_counter()
        jd_chunks = jd_cache.get("chunks", [])
        jd_chunk_embs = np.asarray(jd_cache.get("chunk_embs", []), dtype=np.float32)
        must_skills = jd_extracted.get("must_have_skills", [])
        nice_skills = jd_extracted.get("nice_to_have_skills", [])
        must_embs = np.asarray(jd_cache.get("must_embs", []), dtype=np.float32) if must_skills else None
        nice_embs = np.asarray(jd_cache.get("nice_embs", []), dtype=np.float32) if nice_skills else None

        t_cv_enc0 = time.perf_counter()
        cv_bodies = [strip_cv_pii_for_scoring(c["raw_text"]) for c in candidates]
        cv_chunk_lists = [chunk_text(body, self.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP) for body in cv_bodies]
        flat_chunks = [ch for chs in cv_chunk_lists for ch in chs]
        flat_chunk_embs = self.encode_texts(flat_chunks)
        chunk_bounds = np.cumsum([0] + [len(chs) for chs in cv_chunk_lists])
        cv_chunk_embs_list = [flat_chunk_embs[chunk_bounds[i]:chunk_bounds[i + 1]] for i in range(len(candidates))]
        cv_encode_ms = (time.perf_counter() - t_cv_enc0) * 1000.0

        t_sk_enc0 = time.perf_counter()
        cv_skills_list = [c["extracted"].get("skills", []) for c in candidates]
        flat_skills = [s for skills in cv_skills_list for s in skills]
        flat_skill_embs = self.encode_texts(flat_skills) if flat_skills else np.zeros((0, 384), dtype=np.float32)
        sk_bounds = np.cumsum([0] + [len(s) for s in cv_skills_list])
        cv_skill_embs_list = [
            flat_skill_embs[sk_bounds[i]:sk_bounds[i + 1]] if cv_skills_list[i] else None
            for i in range(len(candidates))
        ]
        skill_encode_ms = (time.perf_counter() - t_sk_enc0) * 1000.0

        t_score0 = time.perf_counter()
        p5, p95 = self.cfg["scaler"]
        tau = float(self.cfg["skill_threshold"])
        rho = float(self.cfg.get("must_ratio", 1.0))
        t1, t2 = float(self.cfg["t1"]), float(self.cfg["t2"])
        weights = self.cfg["weights"]
        jd_years = float(jd_extracted.get("min_years_experience", 0.0))

        outputs: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            t_one0 = time.perf_counter()
            cv_ent = cand["extracted"]
            cv_embs = cv_chunk_embs_list[idx]
            cv_skills = cv_skills_list[idx]
            cv_sk_embs = cv_skill_embs_list[idx]

            s_sem = scale_p5_p95(max_mean_sim(jd_chunk_embs, cv_embs), p5, p95)
            s_skill, miss_count = compute_skill_score(
                must_skills, nice_skills, cv_skills,
                must_skill_embeddings=must_embs,
                nice_skill_embeddings=nice_embs,
                cv_skill_embeddings=cv_sk_embs,
                skill_threshold=tau,
                synonyms=SKILL_SYNONYMS,
            )
            cv_years = float(cv_ent.get("total_years_experience", 0.0))
            s_exp = compute_exp_score(cv_years, jd_years)
            s_edu = compute_edu_score(
                cv_ent.get("highest_degree", "none"),
                jd_extracted.get("min_degree", "none"),
                cv_ent.get("major_group", "unknown"),
                jd_extracted.get("major_group", "unknown"),
            )
            s_cert = compute_cert_score(cv_ent.get("certs", []), jd_extracted.get("required_certs", []))

            criteria = {"sem": round(s_sem, 2), "skill": _r2(s_skill), "exp": _r2(s_exp), "edu": _r2(s_edu), "cert": _r2(s_cert)}
            raw_score = weighted_score({"sem": s_sem, "skill": s_skill, "exp": s_exp, "edu": s_edu, "cert": s_cert}, weights)
            final_score, failed_reasons = apply_must_have(
                raw_score, miss_count, cv_years, jd_years,
                t2=t2, must_total=len(must_skills), must_ratio=rho,
            )
            level = 2 if final_score >= t2 else (1 if final_score >= t1 else 0)
            matched, missing = _skill_breakdown(must_skills, nice_skills, cv_skills, must_embs, nice_embs, cv_sk_embs, tau)
            best_matches = _chunk_evidence(jd_chunks, jd_chunk_embs, cv_chunk_lists[idx], cv_embs)
            elapsed_ms = (time.perf_counter() - t_one0) * 1000.0 + (cv_encode_ms + skill_encode_ms) / max(len(candidates), 1)

            outputs.append({
                "candidate_id": cand["id"],
                "raw_score": round(float(raw_score), 4),
                "final_score": round(float(final_score), 4),
                "level": level,
                "level_label": LEVEL_LABELS[level],
                "criteria": criteria,
                "matched": matched,
                "missing": missing,
                "must_have_failed": failed_reasons,
                "best_matches": best_matches,
                "model_version": self.model_version,
                "elapsed_ms": round(elapsed_ms, 2),
            })

        scoring_ms = (time.perf_counter() - t_score0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        timings = {
            "cv_encode_ms": round(cv_encode_ms, 2),
            "skill_encode_ms": round(skill_encode_ms, 2),
            "scoring_ms": round(scoring_ms, 2),
            "total_ms": round(total_ms, 2),
            "per_cv_ms": round(total_ms / max(len(candidates), 1), 2),
        }
        return outputs, timings


def _r2(val: float | None) -> float | None:
    return None if val is None else round(float(val), 2)


def _skill_breakdown(must_skills, nice_skills, cv_skills, must_embs, nice_embs, cv_sk_embs, tau):
    cv_norm = {normalize_skill(s, SKILL_SYNONYMS) for s in cv_skills}
    matched, missing = [], []
    for group_name, skills, embs in (("must", must_skills, must_embs), ("nice", nice_skills, nice_embs)):
        for i, skill in enumerate(skills):
            emb = embs[i] if embs is not None and i < len(embs) else None
            score = match_single_skill(normalize_skill(skill, SKILL_SYNONYMS), emb, cv_norm, cv_sk_embs, tau)
            item = {"skill": skill, "type": group_name, "score": round(float(score), 3)}
            (matched if score > 0 else missing).append(item)
    return matched, missing


def _chunk_evidence(jd_chunks, jd_embs, cv_chunks, cv_embs, max_items: int = 5) -> List[Dict[str, Any]]:
    if len(jd_chunks) == 0 or len(cv_chunks) == 0 or len(jd_embs) == 0 or len(cv_embs) == 0:
        return []
    sim_mat = np.dot(jd_embs, cv_embs.T)
    evidence = []
    for i, jd_chunk in enumerate(jd_chunks[:max_items]):
        best_j = int(np.argmax(sim_mat[i]))
        evidence.append({
            "jd_chunk_idx": i + 1,
            "jd_excerpt": jd_chunk[:240],
            "cv_excerpt": cv_chunks[best_j][:240],
            "similarity": round(float(sim_mat[i, best_j]), 4),
        })
    return evidence

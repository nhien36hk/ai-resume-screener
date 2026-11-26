from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.loader import load_all, select_split
from src.config import (
    ALL_PAIRS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    MAX_SEQ_LENGTH,
    SKILL_THRESHOLDS,
)
from src.features.skills import SKILL_SYNONYMS
from src.features.text import chunk_text, prepare_finetune_text
from src.scoring import (
    compute_cert_score,
    compute_edu_score,
    compute_exp_score,
    compute_skill_score,
    max_mean_sim,
)


def precompute_all_features(
    df: pd.DataFrame,
    model: Optional[SentenceTransformer] = None,
) -> pd.DataFrame:
    """Xây dựng bảng đặc trưng tiền tính toán cho mọi cặp CV-JD trong split.
    
    Tận dụng tính duy nhất (351 JD, 643 CV) để encode 1 lần duy nhất.
    """
    if model is None:
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        model.max_seq_length = MAX_SEQ_LENGTH

    tokenizer = model.tokenizer

    # 1. Tiền xử lý và encode các JD duy nhất
    unique_jds = {}
    for job_id, group in tqdm(df.groupby("job_id"), desc="Encode JDs"):
        jd_row = group.iloc[0]
        jd_text = jd_row["job_description_text"]
        chunks = chunk_text(jd_text, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)  # JD chunk: giảng viên đã đồng ý
        chunk_embs = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        trunc_emb = model.encode([jd_text],  # E3a: model tự cắt ở max_seq_length = 256 token
                               normalize_embeddings=True, convert_to_numpy=True)[0]

        jd_entities = jd_row.get("jd_entities", {})
        must_skills = jd_entities.get("must_have_skills", [])
        nice_skills = jd_entities.get("nice_to_have_skills", [])
        must_skill_embeddings = model.encode(must_skills, normalize_embeddings=True, convert_to_numpy=True) if must_skills else None
        nice_skill_embeddings = model.encode(nice_skills, normalize_embeddings=True, convert_to_numpy=True) if nice_skills else None

        unique_jds[job_id] = {
            "chunks": chunks,
            "chunk_embs": chunk_embs,
            "trunc_emb": trunc_emb,
            "entities": jd_entities,
            "must_skill_embeddings": must_skill_embeddings,
            "nice_skill_embeddings": nice_skill_embeddings,
        }

    # 2. Tiền xử lý và encode các CV duy nhất
    unique_cvs = {}
    for cv_text, group in tqdm(df.groupby("resume_text"), desc="Encode CVs"):
        cv_row = group.iloc[0]
        cv_body = prepare_finetune_text(cv_text)  # V7: cùng bước bỏ phần đầu CV như khi huấn luyện M5
        chunks = chunk_text(cv_body, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
        chunk_embs = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        trunc_emb = model.encode([cv_body], normalize_embeddings=True, convert_to_numpy=True)[0]

        cv_entities = cv_row.get("cv_entities", {})
        cv_skills = cv_entities.get("skills", [])
        cv_skill_embeddings = model.encode(cv_skills, normalize_embeddings=True, convert_to_numpy=True) if cv_skills else None

        unique_cvs[cv_text] = {
            "chunks": chunks,
            "chunk_embs": chunk_embs,
            "trunc_emb": trunc_emb,
            "entities": cv_entities,
            "cv_skill_embeddings": cv_skill_embeddings,
        }

    # 3. Tính toán các tiêu chí cho từng cặp
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Build Features"):
        job_id = row["job_id"]
        cv_text = row["resume_text"]
        jd_data = unique_jds[job_id]
        cv_data = unique_cvs[cv_text]

        jd_entities = jd_data["entities"]
        cv_entities = cv_data["entities"]

        sim_chunk_200 = max_mean_sim(jd_data["chunk_embs"], cv_data["chunk_embs"])
        sim_truncated_256 = float(np.dot(jd_data["trunc_emb"], cv_data["trunc_emb"]))

        must_skills = jd_entities.get("must_have_skills", [])
        nice_skills = jd_entities.get("nice_to_have_skills", [])
        cv_skills = cv_entities.get("skills", [])

        must_skill_embeddings = jd_data["must_skill_embeddings"]
        nice_skill_embeddings = jd_data["nice_skill_embeddings"]
        cv_skill_embeddings = cv_data["cv_skill_embeddings"]

        skill_scores = {}
        miss_counts = {}
        for skill_threshold in SKILL_THRESHOLDS:
            score_skill, miss_count = compute_skill_score(
                must_skills, nice_skills, cv_skills,
                must_skill_embeddings=must_skill_embeddings,
                nice_skill_embeddings=nice_skill_embeddings,
                cv_skill_embeddings=cv_skill_embeddings,
                skill_threshold=skill_threshold,
                synonyms=SKILL_SYNONYMS,
            )
            skill_scores[f"score_skill_{skill_threshold}"] = score_skill
            miss_counts[f"miss_count_{skill_threshold}"] = miss_count

        cv_years_exp = float(cv_entities.get("total_years_experience", 0.0))
        jd_min_years_exp = float(jd_entities.get("min_years_experience", 0.0))
        score_experience = compute_exp_score(cv_years_exp, jd_min_years_exp)
        has_insufficient_exp = (jd_min_years_exp > 0.0 and cv_years_exp < 0.5 * jd_min_years_exp)

        score_education = compute_edu_score(
            cv_degree=cv_entities.get("highest_degree", "none"),
            jd_degree=jd_entities.get("min_degree", "none"),
            cv_major_group=cv_entities.get("major_group", "unknown"),
            jd_major_group=jd_entities.get("major_group", "unknown"),
        )

        score_certificate = compute_cert_score(
            cv_certs=cv_entities.get("certs", []),
            jd_certs=jd_entities.get("required_certs", []),
        )

        record = {
            "pair_id": row.get("pair_id", ""),
            "job_id": job_id,
            "label": int(row["label"]),
            "sim_chunk_200": sim_chunk_200,
            "sim_truncated_256": sim_truncated_256,
            **skill_scores,
            **miss_counts,
            "total_must_skills": len(must_skills),
            "score_experience": score_experience,
            "score_education": score_education,
            "score_certificate": score_certificate,
            "has_insufficient_exp": has_insufficient_exp,
            "cv_years_exp": cv_years_exp,
            "jd_min_years_exp": jd_min_years_exp,
        }
        rows.append(record)

    return pd.DataFrame(rows)
    

def get_split_features(feat_dir: Path, model: SentenceTransformer, seed: int, scheme: str = "jd") -> Dict[str, pd.DataFrame]:
    """Bảng feature là cache duy nhất: all.parquet dùng chung mọi seed và scheme, cắt theo file index."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    all_parquet = feat_dir / "all.parquet"
    if all_parquet.exists():
        df_feat = pd.read_parquet(all_parquet)
        if "total_must_skills" not in df_feat.columns:
            all_parquet.unlink(missing_ok=True)

    if not all_parquet.exists() or (ALL_PAIRS_PATH.exists() and all_parquet.stat().st_mtime < ALL_PAIRS_PATH.stat().st_mtime):
        print(f"Tính bảng feature toàn bộ -> {all_parquet}")
        precompute_all_features(load_all(), model).to_parquet(all_parquet, index=False)

    df_feat = pd.read_parquet(all_parquet)
    return {name: select_split(df_feat, name, seed, scheme) for name in ("train", "val", "test")}

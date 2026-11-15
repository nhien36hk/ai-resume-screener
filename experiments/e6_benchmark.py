from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL, MAX_SEQ_LENGTH
from src.data.loader import load_all
from src.features.skills import SKILL_SYNONYMS
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
from src.utils import save_json, to_markdown_table


def benchmark_single_pass(jd_row, cv_rows, encoder, percentile_p5, percentile_p95, weights, t1, t2):
    jd_text = jd_row["job_description_text"]
    jd_entities = jd_row.get("jd_entities", {})
    jd_chunks = chunk_text(jd_text, encoder.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
    jd_chunk_embs = encoder.encode(jd_chunks, normalize_embeddings=True, convert_to_numpy=True)

    must_skills = jd_entities.get("must_have_skills", [])
    nice_skills = jd_entities.get("nice_to_have_skills", [])
    must_skill_embeddings = encoder.encode(must_skills, normalize_embeddings=True, convert_to_numpy=True) if must_skills else None
    nice_skill_embeddings = encoder.encode(nice_skills, normalize_embeddings=True, convert_to_numpy=True) if nice_skills else None

    jd_min_years_exp = float(jd_entities.get("min_years_experience", 0.0))
    jd_deg = jd_entities.get("min_degree", "none")
    jd_major = jd_entities.get("major_group", "unknown")
    jd_certs = jd_entities.get("required_certs", [])

    results = []
    for _, cv_row in cv_rows.iterrows():
        cv_text = cv_row["resume_text"]
        cv_entities = cv_row.get("cv_entities", {})

        chunks = chunk_text(cv_text, encoder.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
        chunk_embs = encoder.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        sim = max_mean_sim(jd_chunk_embs, chunk_embs)
        score_semantic = scale_p5_p95(sim, percentile_p5, percentile_p95)

        cv_skills = cv_entities.get("skills", [])
        cv_skill_embeddings = encoder.encode(cv_skills, normalize_embeddings=True, convert_to_numpy=True) if cv_skills else None
        score_skill, miss_count = compute_skill_score(
            must_skills, nice_skills, cv_skills,
            must_skill_embeddings=must_skill_embeddings,
            nice_skill_embeddings=nice_skill_embeddings,
            cv_skill_embeddings=cv_skill_embeddings,
            skill_threshold=0.7, synonyms=SKILL_SYNONYMS,
        )

        cv_years_exp = float(cv_entities.get("total_years_experience", 0.0))
        score_experience = compute_exp_score(cv_years_exp, jd_min_years_exp)
        score_education = compute_edu_score(cv_entities.get("highest_degree", "none"), jd_deg, cv_entities.get("major_group", "unknown"), jd_major)
        score_certificate = compute_cert_score(cv_entities.get("certs", []), jd_certs)

        scores_dict = {"sem": score_semantic, "skill": score_skill, "exp": score_experience, "edu": score_education, "cert": score_certificate}
        score_aggregate = weighted_score(scores_dict, weights)
        score_final, _ = apply_must_have(score_aggregate, miss_count, cv_years_exp, jd_min_years_exp, t2=t2)

        pred = 2 if score_final >= t2 else (1 if score_final >= t1 else 0)
        results.append(pred)

    return results


def run_e6():
    print("=== CHẠY THỰC NGHIỆM E6: BENCHMARK TỐC ĐỘ VÀ ĐỘ TRỄ HỆ THỐNG ===")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Thiết bị thử nghiệm: {device_name}")

    encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    df_all = load_all()
    jd_sample = df_all.iloc[0]
    unique_cvs_df = df_all.drop_duplicates(subset=["resume_text"]).copy()

    percentile_p5, percentile_p95 = 0.30, 0.80
    weights = {"sem": 0.2, "skill": 0.2, "exp": 0.2, "edu": 0.2, "cert": 0.2}
    t1, t2 = 40.0, 70.0

    print("Thực hiện 1 lượt khởi động (Warm-up)...")
    benchmark_single_pass(jd_sample, unique_cvs_df.head(2), encoder, percentile_p5, percentile_p95, weights, t1, t2)

    cv_batches = [1, 10, 50, 100]
    n_repeats = 5
    benchmark_records = []

    for n_cv in cv_batches:
        cv_subset = unique_cvs_df.head(n_cv)
        actual_n = len(cv_subset)
        latencies = []

        for _ in range(n_repeats):
            start = time.perf_counter()
            benchmark_single_pass(jd_sample, cv_subset, encoder, percentile_p5, percentile_p95, weights, t1, t2)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        lat_arr = np.array(latencies)
        mean_sec = float(np.mean(lat_arr))
        p95_sec = float(np.percentile(lat_arr, 95))
        throughput = actual_n / mean_sec

        benchmark_records.append({
            "n_cv": actual_n,
            "mean_sec": mean_sec,
            "p95_sec": p95_sec,
            "throughput_cv_s": throughput,
        })
        print(f"Số CV: {actual_n:3d} | Mean: {mean_sec*1000:7.1f} ms | p95: {p95_sec*1000:7.1f} ms | Throughput: {throughput:5.1f} CV/s")

    headers = ["Số lượng CV", "Thời gian TB (ms)", "Thời gian p95 (ms)", "Thông lượng (CV/s)"]
    rows = [
        [
            r["n_cv"],
            f"{r['mean_sec']*1000:.1f}",
            f"{r['p95_sec']*1000:.1f}",
            f"{r['throughput_cv_s']:.2f}",
        ]
        for r in benchmark_records
    ]
    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir = PROJECT_ROOT / "artifacts" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "e6.json", {"device": device_name, "benchmarks": benchmark_records})
    with open(out_dir / "e6.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E6: Benchmark Hiệu năng ({device_name})\n\n" + md_table + "\n")
    print(f"Đã lưu kết quả vào {out_dir}/e6.json và e6.md")


if __name__ == "__main__":
    run_e6()

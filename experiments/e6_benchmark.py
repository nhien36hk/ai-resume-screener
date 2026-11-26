"""E6 (RQ5): thời gian xử lý 1 JD với 1/10/50/100 CV, tách theo bước, cấu hình lấy từ M4 (E4 seed 0)."""
import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL, MAX_SEQ_LENGTH, RESULTS_DIR
from src.data.loader import load_all
from src.features.skills import SKILL_SYNONYMS
from src.features.text import chunk_text, prepare_finetune_text
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
from src.utils import load_json, save_json, to_markdown_table

BATCH_SIZE = 64
STEPS = ("jd", "cv_encode", "skill_encode", "scoring")


def encode(encoder, texts):
    return encoder.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_single_pass(jd_row, cv_rows, encoder, cfg):
    """Chấm 1 JD với các CV trong cv_rows. Trả (nhãn dự đoán, thời gian từng bước tính bằng giây)."""
    timing = {}
    t = time.perf_counter()
    jd_entities = jd_row.get("jd_entities", {})
    jd_chunk_embs = encode(encoder, chunk_text(jd_row["job_description_text"], encoder.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP))
    must_skills = jd_entities.get("must_have_skills", [])
    nice_skills = jd_entities.get("nice_to_have_skills", [])
    must_embs = encode(encoder, must_skills) if must_skills else None
    nice_embs = encode(encoder, nice_skills) if nice_skills else None
    sync()
    timing["jd"] = time.perf_counter() - t

    # Encode chunk của mọi CV trong một lần gọi theo lô, rồi tách lại theo CV
    t = time.perf_counter()
    cv_chunks = [chunk_text(prepare_finetune_text(txt), encoder.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP) for txt in cv_rows["resume_text"]]
    all_embs = encode(encoder, [c for chunks in cv_chunks for c in chunks])
    bounds = np.cumsum([0] + [len(c) for c in cv_chunks])
    cv_chunk_embs = [all_embs[bounds[i]:bounds[i + 1]] for i in range(len(cv_chunks))]
    sync()
    timing["cv_encode"] = time.perf_counter() - t

    t = time.perf_counter()
    cv_skills = [ent.get("skills", []) for ent in cv_rows["cv_entities"]]
    flat = [s for skills in cv_skills for s in skills]
    flat_embs = encode(encoder, flat) if flat else np.zeros((0, 1))
    bounds = np.cumsum([0] + [len(s) for s in cv_skills])
    cv_skill_embs = [flat_embs[bounds[i]:bounds[i + 1]] if cv_skills[i] else None for i in range(len(cv_skills))]
    sync()
    timing["skill_encode"] = time.perf_counter() - t

    t = time.perf_counter()
    jd_years = float(jd_entities.get("min_years_experience", 0.0))
    p5, p95 = cfg["scaler"]
    preds = []
    for i, cv_entities in enumerate(cv_rows["cv_entities"]):
        score_skill, miss_count = compute_skill_score(
            must_skills, nice_skills, cv_skills[i],
            must_skill_embeddings=must_embs, nice_skill_embeddings=nice_embs, cv_skill_embeddings=cv_skill_embs[i],
            skill_threshold=cfg["skill_threshold"], synonyms=SKILL_SYNONYMS,
        )
        cv_years = float(cv_entities.get("total_years_experience", 0.0))
        scores = {
            "sem": scale_p5_p95(max_mean_sim(jd_chunk_embs, cv_chunk_embs[i]), p5, p95),
            "skill": score_skill,
            "exp": compute_exp_score(cv_years, jd_years),
            "edu": compute_edu_score(cv_entities.get("highest_degree", "none"), jd_entities.get("min_degree", "none"),
                                     cv_entities.get("major_group", "unknown"), jd_entities.get("major_group", "unknown")),
            "cert": compute_cert_score(cv_entities.get("certs", []), jd_entities.get("required_certs", [])),
        }
        final, _ = apply_must_have(weighted_score(scores, cfg["weights"]), miss_count, cv_years, jd_years, t2=cfg["t2"],
                                   must_total=len(must_skills), must_ratio=cfg.get("must_ratio", 1.0))
        preds.append(2 if final >= cfg["t2"] else (1 if final >= cfg["t1"] else 0))
    timing["scoring"] = time.perf_counter() - t
    return preds, timing


def run_e6(config_path: Path = RESULTS_DIR / "jd" / "seed0" / "e4.json", n_repeats: int = 5):
    print("=== CHẠY THỰC NGHIỆM E6: BENCHMARK TỐC ĐỘ VÀ ĐỘ TRỄ HỆ THỐNG ===")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Thiết bị thử nghiệm: {device_name}")
    e4 = load_json(config_path, None)
    if not e4:
        raise FileNotFoundError(f"Chưa có {config_path}: chạy E4 (seed 0, chia jd) trước để lấy cấu hình M4.")
    cfg = e4["config"]
    print(f"Cấu hình M4 từ {config_path}: τ={cfg['skill_threshold']}, ρ={cfg.get('must_ratio', 1.0)}, t=({cfg['t1']:.0f}, {cfg['t2']:.0f})")

    encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    df_all = load_all()
    jd_sample = df_all.iloc[0]
    unique_cvs = df_all.drop_duplicates(subset=["resume_text"])

    print("Thực hiện 1 lượt khởi động (Warm-up)...")
    benchmark_single_pass(jd_sample, unique_cvs.head(2), encoder, cfg)

    records = []
    for n_cv in (1, 10, 50, 100):
        cv_subset = unique_cvs.head(n_cv)
        n = len(cv_subset)
        n_tokens = float(np.mean([len(encoder.tokenizer.encode(prepare_finetune_text(t), add_special_tokens=False)) for t in cv_subset["resume_text"]]))
        runs = [benchmark_single_pass(jd_sample, cv_subset, encoder, cfg)[1] for _ in range(n_repeats)]
        totals = np.array([sum(r.values()) for r in runs])
        per_cv = totals / n
        rec = {
            "n_cv": n, "avg_cv_tokens": n_tokens,
            "total_mean_sec": float(totals.mean()), "total_sd_sec": float(totals.std(ddof=1)),
            "per_cv_mean_sec": float(per_cv.mean()), "per_cv_p95_sec": float(np.percentile(per_cv, 95)),
            "throughput_cv_s": float(n / totals.mean()),
            "steps_mean_sec": {k: float(np.mean([r[k] for r in runs])) for k in STEPS},
        }
        records.append(rec)
        print(f"Số CV: {n:3d} | Tổng: {rec['total_mean_sec']*1000:8.1f} ± {rec['total_sd_sec']*1000:6.1f} ms | "
              f"Mỗi CV: {rec['per_cv_mean_sec']*1000:6.1f} ms (p95 {rec['per_cv_p95_sec']*1000:6.1f}) | {rec['throughput_cv_s']:5.1f} CV/s")

    headers = ["Số CV", "Token CV TB", "Tổng (ms, mean ± sd)", "Mỗi CV TB (ms)", "Mỗi CV p95 (ms)", "Thông lượng (CV/s)",
               "JD (ms)", "Encode CV (ms)", "Encode kỹ năng (ms)", "Chấm điểm (ms)"]
    rows = [[
        r["n_cv"], f"{r['avg_cv_tokens']:.0f}", f"{r['total_mean_sec']*1000:.1f} ± {r['total_sd_sec']*1000:.1f}",
        f"{r['per_cv_mean_sec']*1000:.1f}", f"{r['per_cv_p95_sec']*1000:.1f}", f"{r['throughput_cv_s']:.2f}",
        *[f"{r['steps_mean_sec'][k]*1000:.1f}" for k in STEPS],
    ] for r in records]
    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "e6.json", {"device": device_name, "batch_size": BATCH_SIZE, "n_repeats": n_repeats,
                                    "config_from": str(config_path), "benchmarks": records})
    with open(out_dir / "e6.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E6: Benchmark Hiệu năng ({device_name}, batch {BATCH_SIZE}, {n_repeats} lần lặp)\n\n" + md_table + "\n")
    print(f"Đã lưu kết quả vào {out_dir}/e6.json và e6.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=RESULTS_DIR / "jd" / "seed0" / "e4.json", help="e4.json chứa cấu hình M4.")
    args = parser.parse_args()
    run_e6(args.config)

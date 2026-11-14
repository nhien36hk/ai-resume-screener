import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import pandas as pd
from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import GuidedDecodingParams
    GUIDED_ARG = "guided_decoding"
except ImportError:
    from vllm.sampling_params import StructuredOutputsParams as GuidedDecodingParams
    GUIDED_ARG = "structured_outputs"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CV_CACHE_PATH,
    DEFAULT_VLLM_MODEL,
    JD_CACHE_PATH,
    MAJOR_GROUPS,
    PROCESSED_DIR,
)
from src.features.text import number_segments
from src.utils import compute_text_hash, load_json, save_json


CV_MAJOR_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
        "majors": {"type": "array", "items": {"type": "string"}},
        "major_group": {"type": "string", "enum": MAJOR_GROUPS}
    },
    "required": ["evidence_ids", "majors", "major_group"]
}

JD_MAJOR_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
        "required_majors": {"type": "array", "items": {"type": "string"}},
        "major_group": {"type": "string", "enum": MAJOR_GROUPS},
        "accepts_related_field": {"type": "boolean"}
    },
    "required": ["evidence_ids", "required_majors", "major_group", "accepts_related_field"]
}

CV_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia phân tích hồ sơ ứng viên (CV/Resume).
Nhiệm vụ: Tìm thông tin học vấn, bằng cấp và chuyên ngành học trong văn bản CV đã được đánh số thứ tự [1], [2], [3]...
Quy tắc:
1. 'evidence_ids': Liệt kê tối đa 3 số thứ tự các đoạn chứa thông tin bằng cấp/ngành học. Nếu không có học vấn, trả về [].
2. 'majors': Tên chuyên ngành viết thường (ví dụ: 'computer science', 'accounting').
3. 'major_group': Chọn ĐÚNG MỘT nhóm trong danh sách enum. Nếu CV không ghi ngành học cụ thể, bắt buộc chọn 'unknown'.
4. Trả về đúng định dạng JSON được yêu cầu."""

JD_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia phân tích yêu cầu tuyển dụng (Job Description).
Nhiệm vụ: Tìm yêu cầu về bằng cấp, chuyên ngành đào tạo trong văn bản JD đã được đánh số thứ tự [1], [2], [3]...
Quy tắc:
1. 'evidence_ids': Liệt kê tối đa 3 số thứ tự các đoạn yêu cầu ngành học. Nếu JD không yêu cầu ngành, trả về [].
2. 'required_majors': Danh sách ngành yêu cầu viết thường.
3. 'major_group': Nhóm ngành tương ứng trong enum. Nếu JD không yêu cầu ngành, chọn 'unknown'.
4. 'accepts_related_field': Đặt là true nếu JD chấp nhận 'hoặc ngành liên quan' ('or related field').
5. Trả về đúng định dạng JSON được yêu cầu."""


CV_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Văn bản CV đã đánh số:\n"
            "[1] Summary: Software Engineer with 4 years experience building backend APIs.\n"
            "[2] Experience: Software Developer at Tech Corp (2021 - Present).\n"
            "[3] Education\n"
            "[4] Bachelor of Science in Computer Science, University of California (2017 - 2021).\n"
            "[5] Skills: Python, SQL, Docker, FastAPI.\n\n"
            "Trích xuất JSON học vấn theo đúng schema:"
        )
    },
    {
        "role": "assistant",
        "content": '{"evidence_ids": [4], "majors": ["computer science"], "major_group": "computer_science"}'
    },
    {
        "role": "user",
        "content": (
            "Văn bản CV đã đánh số:\n"
            "[1] Profile: Retail Sales Associate with 3 years of customer service.\n"
            "[2] Education\n"
            "[3] High School Diploma - Lincoln High School (Graduated 2019).\n"
            "[4] Skills: Cash handling, POS systems, Communication.\n\n"
            "Trích xuất JSON học vấn theo đúng schema:"
        )
    },
    {
        "role": "assistant",
        "content": '{"evidence_ids": [3], "majors": [], "major_group": "unknown"}'
    }
]

JD_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Văn bản JD đã đánh số:\n"
            "[1] Role: Senior Financial Analyst\n"
            "[2] Responsibilities: Prepare monthly financial statements and variance analysis.\n"
            "[3] Requirements:\n"
            "[4] Bachelor degree in Finance, Accounting, or a closely related field.\n"
            "[5] 3+ years experience in corporate finance.\n\n"
            "Trích xuất JSON yêu cầu ngành theo đúng schema:"
        )
    },
    {
        "role": "assistant",
        "content": (
            '{"evidence_ids": [4], "required_majors": ["finance", "accounting"], '
            '"major_group": "accounting_finance", "accepts_related_field": true}'
        )
    },
    {
        "role": "user",
        "content": (
            "Văn bản JD đã đánh số:\n"
            "[1] Role: Customer Success Representative\n"
            "[2] Requirements:\n"
            "[3] High School Diploma or equivalent required; Bachelor degree in any major is a plus.\n"
            "[4] Excellent communication skills.\n\n"
            "Trích xuất JSON yêu cầu ngành theo đúng schema:"
        )
    },
    {
        "role": "assistant",
        "content": '{"evidence_ids": [3], "required_majors": [], "major_group": "unknown", "accepts_related_field": false}'
    }
]


def build_cv_prompt(numbered_cv: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CV_SYSTEM_PROMPT},
        *CV_FEW_SHOT,
        {"role": "user", "content": f"Văn bản CV đã đánh số:\n{numbered_cv}\n\nTrích xuất JSON học vấn theo đúng schema:"}
    ]


def build_jd_prompt(numbered_jd: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": JD_SYSTEM_PROMPT},
        *JD_FEW_SHOT,
        {"role": "user", "content": f"Văn bản JD đã đánh số:\n{numbered_jd}\n\nTrích xuất JSON yêu cầu ngành theo đúng schema:"}
    ]


def run_vllm_batch(
    model_name: str,
    jd_cache_path: Path = JD_CACHE_PATH,
    cv_cache_path: Path = CV_CACHE_PATH,
    sample_limit: Optional[int] = None
) -> None:

    all_dfs = []
    for split in ["train.parquet", "val.parquet", "test.parquet"]:
        p = PROCESSED_DIR / split
        if p.exists():
            all_dfs.append(pd.read_parquet(p))
    if not all_dfs:
        print(f"[LỖI] Không tìm thấy dữ liệu parquet trong {PROCESSED_DIR}")
        return
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f">>> Đã nạp tổng cộng {len(df_all):,} cặp CV-JD.")

    jd_cache = load_json(jd_cache_path)
    cv_cache = load_json(cv_cache_path)

    unique_jds_df = df_all[["job_id", "job_description_text"]].drop_duplicates(subset=["job_id"])
    needed_jds = []
    for _, row in unique_jds_df.iterrows():
        j_id = str(row["job_id"])
        has_entities = j_id in jd_cache and "min_degree" in jd_cache[j_id]
        if has_entities and "major_group" not in jd_cache[j_id]:
            needed_jds.append((j_id, row["job_description_text"]))

    unique_cvs_series = df_all["resume_text"].drop_duplicates()
    needed_cvs = []
    for cv_text in unique_cvs_series:
        c_hash = compute_text_hash(cv_text)
        has_entities = c_hash in cv_cache and "skills" in cv_cache[c_hash]
        if has_entities and "major_group" not in cv_cache[c_hash]:
            needed_cvs.append((c_hash, cv_text))

    print("=" * 70)
    print("KẾ HOẠCH BỔ SUNG MAJOR & PHỤC HỒI CACHE:")
    print(f"- Số JD cần trích xuất major : {len(needed_jds)} / {len(unique_jds_df)}")
    print(f"- Số CV cần trích xuất major : {len(needed_cvs)} / {len(unique_cvs_series)}")
    print("=" * 70)

    if sample_limit:
        needed_jds = needed_jds[:sample_limit]
        needed_cvs = needed_cvs[:sample_limit]
        print(f">>> [CHẾ ĐỘ TEST] Giới hạn xử lý tối đa {sample_limit} mẫu.")

    if not needed_jds and not needed_cvs:
        print(">>> Toàn bộ JD và CV đã có đầy đủ trường major_group! Không cần chạy thêm.")
        return

    print(f"\n>>> Đang khởi tạo vLLM Engine với model: {model_name}...")
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        gpu_memory_utilization=0.88,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        seed=42,
    )

    if needed_jds:
        print(f"\n[PHASE A] Đang trích xuất major cho {len(needed_jds)} JD...")
        jd_prompts = []
        jd_segments_map = {}

        for j_id, text in needed_jds:
            numbered_text, segs = number_segments(text)
            jd_segments_map[j_id] = segs
            jd_prompts.append(build_jd_prompt(numbered_text))

        jd_params = SamplingParams(
            temperature=0.0,
            max_tokens=150,
            **{GUIDED_ARG: GuidedDecodingParams(json=JD_MAJOR_SCHEMA)}
        )

        t0 = time.time()
        jd_outputs = llm.chat(
            messages=jd_prompts,
            sampling_params=jd_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": False}
        )
        t_jd = time.time() - t0
        print(f"-> Hoàn thành Phase A trong {t_jd:.2f}s ({len(needed_jds)/max(t_jd, 0.001):.1f} JD/s)")

        for (j_id, _), out in zip(needed_jds, jd_outputs):
            raw_json = out.outputs[0].text
            try:
                res = json.loads(raw_json)
                segs = jd_segments_map[j_id]
                ev_ids = res.get("evidence_ids", [])
                evidence_texts = [segs[i - 1] for i in ev_ids if 1 <= i <= len(segs)]
                
                if j_id not in jd_cache:
                    jd_cache[j_id] = {}
                jd_cache[j_id].update({
                    "required_majors": res.get("required_majors", []),
                    "major_group": res.get("major_group", "unknown"),
                    "accepts_related_field": bool(res.get("accepts_related_field", False)),
                    "major_evidence": " | ".join(evidence_texts)
                })
            except Exception as e:
                print(f"[CẢNH BÁO] Không parse được JSON cho JD {j_id}: {e}")

        save_json(jd_cache_path, jd_cache)
        print(f">>> Đã cập nhật và lưu cache JD tại: {jd_cache_path}")

    if needed_cvs:
        print(f"\n[PHASE B] Đang trích xuất major cho {len(needed_cvs)} CV...")
        cv_prompts = []
        cv_segments_map = {}

        for c_hash, text in needed_cvs:
            numbered_text, segs = number_segments(text)
            cv_segments_map[c_hash] = segs
            cv_prompts.append(build_cv_prompt(numbered_text))

        cv_params = SamplingParams(
            temperature=0.0,
            max_tokens=150,
            **{GUIDED_ARG: GuidedDecodingParams(json=CV_MAJOR_SCHEMA)}
        )

        t0 = time.time()
        cv_outputs = llm.chat(
            messages=cv_prompts,
            sampling_params=cv_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": False}
        )
        t_cv = time.time() - t0
        print(f"-> Hoàn thành Phase B trong {t_cv:.2f}s ({len(needed_cvs)/max(t_cv, 0.001):.1f} CV/s)")

        for (c_hash, _), out in zip(needed_cvs, cv_outputs):
            raw_json = out.outputs[0].text
            try:
                res = json.loads(raw_json)
                segs = cv_segments_map[c_hash]
                ev_ids = res.get("evidence_ids", [])
                evidence_texts = [segs[i - 1] for i in ev_ids if 1 <= i <= len(segs)]

                if c_hash not in cv_cache:
                    cv_cache[c_hash] = {}
                cv_cache[c_hash].update({
                    "majors": res.get("majors", []),
                    "major_group": res.get("major_group", "unknown"),
                    "major_evidence": " | ".join(evidence_texts)
                })
            except Exception as e:
                print(f"[CẢNH BÁO] Không parse được JSON cho CV {c_hash[:8]}: {e}")

        save_json(cv_cache_path, cv_cache)
        print(f">>> Đã cập nhật và lưu cache CV tại: {cv_cache_path}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT BỔ SUNG MAJOR CHO TOÀN BỘ DATASET!")
    print(f"- Tổng số JD có major trong cache: {len(jd_cache)}")
    print(f"- Tổng số CV có major trong cache: {len(cv_cache)}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất major cho CV/JD bằng vLLM (Qwen3-14B-FP8).")
    default_model = "/tmp/Qwen3-14B-FP8" if Path("/tmp/Qwen3-14B-FP8").exists() else DEFAULT_VLLM_MODEL
    parser.add_argument("--model", type=str, default=default_model, help="Tên hoặc đường dẫn model Qwen FP8.")
    parser.add_argument("--jd-cache", type=Path, default=JD_CACHE_PATH, help="Đường dẫn file cache JD.")
    parser.add_argument("--cv-cache", type=Path, default=CV_CACHE_PATH, help="Đường dẫn file cache CV.")
    parser.add_argument("--test-sample", type=int, default=None, help="Chạy thử nghiệm một số lượng mẫu nhỏ (VD: 20).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vllm_batch(
        model_name=args.model,
        jd_cache_path=args.jd_cache,
        cv_cache_path=args.cv_cache,
        sample_limit=args.test_sample
    )

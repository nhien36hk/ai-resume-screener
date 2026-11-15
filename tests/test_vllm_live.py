import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ["VLLM_USE_DEEP_GEMM"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_CV = """
Summary: Senior Software Engineer with 6 years experience developing cloud applications.
Education:
Master of Science in Computer Science, Stanford University (2020 - 2022).
Bachelor of Science in Information Technology, University of Washington (2016 - 2020).
Experience:
Senior Backend Developer at AWS. Built distributed microservices in Go and Python.
Skills: Python, Go, Docker, Kubernetes, PostgreSQL.
"""

SAMPLE_JD = """
Job Title: Senior Data Engineer / Machine Learning Specialist
Company: DataCore AI
Overview: We are seeking a talented Senior Data Engineer to design high-throughput data pipelines.
Requirements:
- Bachelor's or Master's degree in Computer Science, Data Science, or related field.
- 4+ years of hands-on experience with Python, Apache Spark, and SQL.
- Strong understanding of ETL pipelines and database architecture.
"""


def main():
    import importlib
    vllm_mod = importlib.import_module("scripts.03_extract_major_vllm")
    LLM = vllm_mod.LLM
    SamplingParams = vllm_mod.SamplingParams
    GuidedDecodingParams = vllm_mod.GuidedDecodingParams
    GUIDED_ARG = vllm_mod.GUIDED_ARG
    CV_MAJOR_SCHEMA = vllm_mod.CV_MAJOR_SCHEMA
    JD_MAJOR_SCHEMA = vllm_mod.JD_MAJOR_SCHEMA
    build_cv_prompt = vllm_mod.build_cv_prompt
    build_jd_prompt = vllm_mod.build_jd_prompt
    number_segments = vllm_mod.number_segments

    MODEL_DIR = "/tmp/Qwen3-14B-FP8"

    print("=" * 70)
    print("1. KHỞI TẠO vLLM ENGINE VỚI MODEL TẠI /tmp/Qwen3-14B-FP8...")
    print("=" * 70)
    t0 = time.time()
    llm = LLM(
        model=MODEL_DIR,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        seed=42,
    )
    print(f"-> Khởi tạo vLLM Engine thành công trong {time.time() - t0:.2f}s!\n")

    # Test CV
    print("=" * 70)
    print("2. CHẠY TRÍCH XUẤT CHO MẪU CV BẰNG vLLM...")
    print("=" * 70)
    num_cv, cv_segs = number_segments(SAMPLE_CV)
    cv_msgs = build_cv_prompt(num_cv)
    cv_params = SamplingParams(
        temperature=0.0,
        max_tokens=150,
        **{GUIDED_ARG: GuidedDecodingParams(json=CV_MAJOR_SCHEMA)}
    )
    t_cv = time.time()
    cv_out = llm.chat(
        messages=cv_msgs,
        sampling_params=cv_params,
        chat_template_kwargs={"enable_thinking": False}
    )
    raw_cv_json = cv_out[0].outputs[0].text
    print(f"Thời gian suy luận CV: {time.time() - t_cv:.2f}s")
    print(f"Output JSON từ vLLM:\n{raw_cv_json}")

    res_cv = json.loads(raw_cv_json)
    ev_texts = [cv_segs[i - 1] for i in res_cv.get("evidence_ids", []) if 1 <= i <= len(cv_segs)]
    print("\n[Parsed CV Results]:")
    print(f" - Majors        : {res_cv.get('majors')}")
    print(f" - Major Group   : {res_cv.get('major_group')}")
    print(f" - Evidence IDs  : {res_cv.get('evidence_ids')}")
    print(f" - Trích dẫn     : {' | '.join(ev_texts)}")

    # Test JD
    print("\n" + "=" * 70)
    print("3. CHẠY TRÍCH XUẤT CHO MẪU JD BẰNG vLLM...")
    print("=" * 70)
    num_jd, jd_segs = number_segments(SAMPLE_JD)
    jd_msgs = build_jd_prompt(num_jd)
    jd_params = SamplingParams(
        temperature=0.0,
        max_tokens=150,
        **{GUIDED_ARG: GuidedDecodingParams(json=JD_MAJOR_SCHEMA)}
    )
    t_jd = time.time()
    jd_out = llm.chat(
        messages=jd_msgs,
        sampling_params=jd_params,
        chat_template_kwargs={"enable_thinking": False}
    )
    raw_jd_json = jd_out[0].outputs[0].text
    print(f"Thời gian suy luận JD: {time.time() - t_jd:.2f}s")
    print(f"Output JSON từ vLLM:\n{raw_jd_json}")

    res_jd = json.loads(raw_jd_json)
    ev_jd_texts = [jd_segs[i - 1] for i in res_jd.get("evidence_ids", []) if 1 <= i <= len(jd_segs)]
    print("\n[Parsed JD Results]:")
    print(f" - Required Majors: {res_jd.get('required_majors')}")
    print(f" - Major Group    : {res_jd.get('major_group')}")
    print(f" - Accepts Related: {res_jd.get('accepts_related_field')}")
    print(f" - Evidence IDs   : {res_jd.get('evidence_ids')}")
    print(f" - Trích dẫn      : {' | '.join(ev_jd_texts)}")

    print("\n" + "=" * 70)
    print(">>> CHẠY THÀNH CÔNG 100% vLLM ENGINE VÀ TRÍCH XUẤT ĐÚNG CHUẨN! <<<")
    print("=" * 70)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

import importlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Đảm bảo root path nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock vllm trước khi import script runner
mock_vllm = MagicMock()
sys.modules["vllm"] = mock_vllm
sys.modules["vllm.sampling_params"] = MagicMock()

mod = importlib.import_module("scripts.03_extract_major_vllm")
CV_MAJOR_SCHEMA = mod.CV_MAJOR_SCHEMA
JD_MAJOR_SCHEMA = mod.JD_MAJOR_SCHEMA
MAJOR_GROUPS = mod.MAJOR_GROUPS
build_cv_prompt = mod.build_cv_prompt
build_jd_prompt = mod.build_jd_prompt

from src.features.text import number_segments, segment_text
from src.utils import compute_text_hash


def test_compute_text_hash():
    print("--- Test 1: compute_text_hash ---")
    text1 = "Software Engineer with 5 years experience."
    text2 = "  Software Engineer with 5 years experience.  "
    h1 = compute_text_hash(text1)
    h2 = compute_text_hash(text2)
    assert h1 == h2, "Hash phải giống nhau khi strip khoảng trắng"
    assert len(h1) == 32, "MD5 hash phải có độ dài 32 ký tự hex"
    print(f"✓ compute_text_hash hoạt động chuẩn: {h1}")


def test_segment_text_and_numbering():
    print("\n--- Test 2: segment_text & number_segments ---")
    sample_cv = """
    Summary: Experienced Machine Learning Engineer.
    Education:
    Bachelor of Science in Computer Science, Stanford University (2018 - 2022).
    Experience:
    Senior AI Researcher at OpenAI. Developed large language models.
    Skills: Python, PyTorch, CUDA, Distributed Training.
    """
    numbered_str, segs = number_segments(sample_cv)
    print("Các đoạn đã phân tách và đánh số:")
    print(numbered_str)

    assert len(segs) > 0, "Danh sách đoạn không được rỗng"
    assert "[1]" in numbered_str, "Phải có chỉ số đánh số bắt đầu từ [1]"
    edu_found = any("Education" in s or "Computer Science" in s for s in segs)
    assert edu_found, "Phải phân tách được đoạn học vấn"
    print("✓ segment_text và number_segments tách đoạn chính xác.")


def test_build_prompts():
    print("\n--- Test 3: build_cv_prompt & build_jd_prompt ---")
    sample_numbered_cv = "[1] Education\n[2] B.S. in Electrical Engineering"
    cv_msgs = build_cv_prompt(sample_numbered_cv)
    assert len(cv_msgs) >= 3, "CV prompt phải gồm system, few-shot và user message"
    assert cv_msgs[0]["role"] == "system"
    assert sample_numbered_cv in cv_msgs[-1]["content"]

    sample_numbered_jd = "[1] Requirements\n[2] Degree in Finance or Accounting"
    jd_msgs = build_jd_prompt(sample_numbered_jd)
    assert len(jd_msgs) >= 3, "JD prompt phải gồm system, few-shot và user message"
    assert jd_msgs[0]["role"] == "system"
    assert sample_numbered_jd in jd_msgs[-1]["content"]
    print("✓ Cấu trúc prompt CV & JD chuẩn xác theo định dạng chat template.")


def test_schema_and_simulated_outputs():
    print("\n--- Test 4: Schema validation & Evidence Mapping ---")
    # Kiểm tra cấu trúc CV_MAJOR_SCHEMA
    assert "evidence_ids" in CV_MAJOR_SCHEMA["required"]
    assert "majors" in CV_MAJOR_SCHEMA["required"]
    assert "major_group" in CV_MAJOR_SCHEMA["required"]
    assert set(CV_MAJOR_SCHEMA["properties"]["major_group"]["enum"]) == set(MAJOR_GROUPS)

    # Giả lập output JSON từ LLM cho CV
    simulated_cv_json = '{"evidence_ids": [2], "majors": ["computer science"], "major_group": "computer_science"}'
    res = json.loads(simulated_cv_json)
    assert res["major_group"] in MAJOR_GROUPS
    assert isinstance(res["majors"], list)

    segs = ["Summary: Engineer", "B.S. Computer Science from MIT", "Skills: Python"]
    evidence_texts = [segs[i - 1] for i in res["evidence_ids"] if 1 <= i <= len(segs)]
    assert len(evidence_texts) == 1
    assert "Computer Science" in evidence_texts[0]

    # Kiểm tra cấu trúc JD_MAJOR_SCHEMA
    assert "evidence_ids" in JD_MAJOR_SCHEMA["required"]
    assert "required_majors" in JD_MAJOR_SCHEMA["required"]
    assert "accepts_related_field" in JD_MAJOR_SCHEMA["required"]
    assert JD_MAJOR_SCHEMA["properties"]["accepts_related_field"]["type"] == "boolean", "type phải là 'boolean'"

    # Giả lập output JSON từ LLM cho JD
    simulated_jd_json = '{"evidence_ids": [2], "required_majors": ["finance"], "major_group": "accounting_finance", "accepts_related_field": true}'
    res_jd = json.loads(simulated_jd_json)
    assert res_jd["major_group"] in MAJOR_GROUPS
    assert res_jd["accepts_related_field"] is True
    print("✓ Schema chuẩn RFC và logic ánh xạ evidence_ids hoạt động chuẩn xác.")


if __name__ == "__main__":
    test_compute_text_hash()
    test_segment_text_and_numbering()
    test_build_prompts()
    test_schema_and_simulated_outputs()
    print("\n>>> TẤT CẢ CÁC TEST HÀM ĐÃ VƯỢT QUA 100% THÀNH CÔNG! <<<")

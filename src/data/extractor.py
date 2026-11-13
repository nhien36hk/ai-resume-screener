import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import requests

from src.config import MAJOR_GROUPS

load_dotenv()

DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_MODEL_NAME = DEFAULT_NVIDIA_MODEL
DEFAULT_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("API_NVIDIA", ""))

DEGREE_LEVELS = ["none", "high_school", "associate", "bachelor", "master", "phd"]

JD_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia bóc tách yêu cầu tuyển dụng (Job Description - JD) cho hệ thống ATS.
Nhiệm vụ: Bóc tách thông tin JD thành ĐÚNG MỘT khối JSON theo cấu trúc sau:
{
  "job_title": string,
  "must_have_skills": [string],
  "nice_to_have_skills": [string],
  "min_years_experience": float,
  "min_degree": "none" | "high_school" | "associate" | "bachelor" | "master" | "phd",
  "required_certs": [string],
  "required_majors": [string],
  "major_group": string,
  "accepts_related_field": boolean
}

Quy tắc:
1. skills: Viết thường toàn bộ (lowercase), từ khóa nguyên tử (vd: 'python', 'aws').
2. min_years_experience: Số thực (float). Nếu là khoảng '3-5 năm' lấy cận dưới 3.0. Không yêu cầu ghi 0.0.
3. min_degree: Chỉ chọn 1 trong: ['none', 'high_school', 'associate', 'bachelor', 'master', 'phd'].
4. major_group: Chọn 1 trong danh sách chuẩn (hoặc 'unknown' nếu không yêu cầu).
5. accepts_related_field: true nếu JD chấp nhận 'or related field'/'ngành liên quan'.
6. Chỉ xuất duy nhất một khối JSON hợp lệ."""

JD_FEW_SHOT_USER = """Job Title: Senior Backend Engineer
Requirements:
- At least 4 years of experience building scalable backend services.
- Proficiency in Python, Django, and PostgreSQL is mandatory.
- Experience with AWS Cloud and Docker containers.
- Bachelor degree in Computer Science or related field.
- Nice to have: Experience with Kubernetes, Redis, and AWS Certified Solutions Architect."""

JD_FEW_SHOT_ASSISTANT = """{
  "job_title": "Senior Backend Engineer",
  "must_have_skills": ["python", "django", "postgresql", "backend"],
  "nice_to_have_skills": ["aws", "docker", "kubernetes", "redis"],
  "min_years_experience": 4.0,
  "min_degree": "bachelor",
  "required_certs": ["AWS Certified Solutions Architect"],
  "required_majors": ["computer science"],
  "major_group": "computer_science",
  "accepts_related_field": true
}"""

CV_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia phân tích hồ sơ ứng viên (Resume/CV) cho hệ thống ATS.
Nhiệm vụ: Trích xuất một khối JSON hợp lệ theo đúng cấu trúc sau:
{
  "candidate_title": string,
  "skills": [string],
  "total_years_experience": float,
  "highest_degree": "none" | "high_school" | "associate" | "bachelor" | "master" | "phd",
  "certs": [string],
  "majors": [string],
  "major_group": string
}

Quy tắc:
1. skills: Viết thường toàn bộ (lowercase), chuẩn hóa tên chuẩn.
2. total_years_experience: Số năm kinh nghiệm làm việc thực tế (float). Fresher ghi 0.0.
3. highest_degree: Bậc học cao nhất trong: ['none', 'high_school', 'associate', 'bachelor', 'master', 'phd'].
4. major_group: Chọn 1 trong danh sách chuẩn (hoặc 'unknown' nếu không ghi ngành).
5. Chỉ xuất duy nhất một khối JSON hợp lệ."""

CV_FEW_SHOT_USER = """John Doe - Backend Developer
Summary: Backend engineer with 3.5 years of industry experience developing REST APIs in Python and Go.
Experience:
- Software Engineer at TechCorp (2021 - Present): Developed microservices using Python, FastAPI, Docker, and PostgreSQL on AWS.
Education:
- Bachelor of Science in Information Technology, University of Science (2017 - 2021).
Certifications:
- AWS Certified Developer Associate (2022)
Skills: Python, FastAPI, Go, PostgreSQL, Docker, AWS, Git, Redis."""

CV_FEW_SHOT_ASSISTANT = """{
  "candidate_title": "Backend Developer",
  "skills": ["python", "fastapi", "go", "postgresql", "docker", "aws", "git", "redis"],
  "total_years_experience": 3.5,
  "highest_degree": "bachelor",
  "certs": ["AWS Certified Developer Associate"],
  "majors": ["information technology"],
  "major_group": "information_technology"
}"""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if json_block:
        cleaned = json_block.group(1).strip()
    else:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx:end_idx + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def call_nvidia_llm(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    timeout: int = 120,
) -> str:
    if not api_key:
        api_key = DEFAULT_API_KEY
    if not api_key:
        raise ValueError("Chưa cấu hình API Key. Hãy thêm NVIDIA_API_KEY vào tệp .env.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(DEFAULT_API_URL, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"NVIDIA API Error {response.status_code}: {response.text}")

    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]

    content = message.get("content")
    if not content:
        content = message.get("reasoning_content", "")
    return content


def extract_jd_entities(
    jd_text: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": JD_SYSTEM_PROMPT},
        {"role": "user", "content": JD_FEW_SHOT_USER},
        {"role": "assistant", "content": JD_FEW_SHOT_ASSISTANT},
        {"role": "user", "content": f"Văn bản JD cần bóc tách:\n\"\"\"\n{jd_text}\n\"\"\""}
    ]

    raw_output = call_nvidia_llm(messages, api_key=api_key, model_name=model_name)
    parsed = clean_json_response(raw_output)

    major_grp = parsed.get("major_group", "unknown")
    if major_grp not in MAJOR_GROUPS:
        major_grp = "unknown"

    return {
        "job_title": parsed.get("job_title", ""),
        "must_have_skills": [s.lower().strip() for s in parsed.get("must_have_skills", []) if isinstance(s, str)],
        "nice_to_have_skills": [s.lower().strip() for s in parsed.get("nice_to_have_skills", []) if isinstance(s, str)],
        "min_years_experience": float(parsed.get("min_years_experience", 0.0)),
        "min_degree": parsed.get("min_degree", "none").lower(),
        "required_certs": parsed.get("required_certs", []),
        "required_majors": [m.lower().strip() for m in parsed.get("required_majors", []) if isinstance(m, str)],
        "major_group": major_grp,
        "accepts_related_field": bool(parsed.get("accepts_related_field", False)),
    }


def extract_cv_entities(
    cv_text: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": CV_SYSTEM_PROMPT},
        {"role": "user", "content": CV_FEW_SHOT_USER},
        {"role": "assistant", "content": CV_FEW_SHOT_ASSISTANT},
        {"role": "user", "content": f"Văn bản CV cần bóc tách:\n\"\"\"\n{cv_text}\n\"\"\""}
    ]

    raw_output = call_nvidia_llm(messages, api_key=api_key, model_name=model_name)
    parsed = clean_json_response(raw_output)

    major_grp = parsed.get("major_group", "unknown")
    if major_grp not in MAJOR_GROUPS:
        major_grp = "unknown"

    return {
        "candidate_title": parsed.get("candidate_title", ""),
        "skills": [s.lower().strip() for s in parsed.get("skills", []) if isinstance(s, str)],
        "total_years_experience": float(parsed.get("total_years_experience", 0.0)),
        "highest_degree": parsed.get("highest_degree", "none").lower(),
        "certs": parsed.get("certs", []),
        "majors": [m.lower().strip() for m in parsed.get("majors", []) if isinstance(m, str)],
        "major_group": major_grp,
    }


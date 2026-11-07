# Level 1: Thư viện chuẩn Python (Standard Library)
import json
import os
import re
from typing import Any, Dict, List, Optional

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import requests

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL_NAME = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)

DEGREE_LEVELS = ["none", "high_school", "associate", "bachelor", "master", "phd"]

# System prompt và Few-shot mẫu cho Job Description (JD)
JD_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia bóc tách yêu cầu tuyển dụng (Job Description - JD) cho hệ thống ATS.
Nhiệm vụ của bạn là đọc kỹ văn bản JD và trích xuất thông tin thành ĐÚNG MỘT khối JSON theo cấu trúc sau:
{
  "job_title": string,
  "must_have_skills": [string],
  "nice_to_have_skills": [string],
  "min_years_experience": float,
  "min_degree": "none" | "high_school" | "associate" | "bachelor" | "master" | "phd",
  "required_certs": [string]
}

Quy tắc chuẩn hóa nghiêm ngặt:
1. skills: Viết thường toàn bộ (lowercase), tách nhỏ thành các từ khóa nguyên tử (ví dụ: "python", "fastapi", "docker", "aws").
2. must_have_skills: Kỹ năng bắt buộc phải có, cốt lõi của công việc.
3. nice_to_have_skills: Kỹ năng ưu tiên, điểm cộng (preferred, bonus).
4. min_years_experience: Số năm kinh nghiệm tối thiểu dạng số thực (float). Nếu là khoảng "3-5 năm", lấy cận dưới 3.0. Không yêu cầu ghi 0.0.
5. min_degree: Chỉ chọn 1 giá trị trong tập ["none", "high_school", "associate", "bachelor", "master", "phd"].
6. required_certs: Danh sách tên chứng chỉ yêu cầu (nếu không có thì trả về list rỗng []).
7. Chỉ trả về duy nhất khối JSON hợp lệ, KHÔNG có lời chào hay giải thích ngoài lề."""

JD_FEW_SHOT_USER = """Job Title: Senior Backend Engineer
Requirements:
- At least 4 years of experience building scalable backend services.
- Proficiency in Python, Django, and PostgreSQL is mandatory.
- Experience with AWS Cloud and Docker containers.
- Bachelor degree in Computer Science or equivalent.
- Nice to have: Experience with Kubernetes, Redis, and AWS Certified Solutions Architect."""

JD_FEW_SHOT_ASSISTANT = """{
  "job_title": "Senior Backend Engineer",
  "must_have_skills": ["python", "django", "postgresql", "backend"],
  "nice_to_have_skills": ["aws", "docker", "kubernetes", "redis"],
  "min_years_experience": 4.0,
  "min_degree": "bachelor",
  "required_certs": ["AWS Certified Solutions Architect"]
}"""

# System prompt và Few-shot mẫu cho Resume (CV)
CV_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia phân tích hồ sơ ứng viên (Resume/CV) cho hệ thống ATS.
Hãy trích xuất nhanh và xuất TRỰC TIẾP một khối JSON hợp lệ theo đúng cấu trúc sau:
{
  "candidate_title": string,
  "skills": [string],
  "total_years_experience": float,
  "highest_degree": "none" | "high_school" | "associate" | "bachelor" | "master" | "phd",
  "certs": [string]
}

Quy tắc chuẩn hóa nghiêm ngặt:
1. skills: Viết thường toàn bộ (lowercase), chuẩn hóa tên chuẩn (ví dụ: "reactjs" -> "react", "golang" -> "go", "amazon web services" -> "aws").
2. total_years_experience: Tổng số năm kinh nghiệm làm việc thực tế tính theo dạng số thực (float). Fresher/mới ra trường ghi 0.0.
3. highest_degree: Bậc học cao nhất đạt được, chỉ chọn 1 trong: ["none", "high_school", "associate", "bachelor", "master", "phd"].
4. certs: Các chứng chỉ thực tế ứng viên sở hữu hoặc đã đạt được (list rỗng [] nếu không có).
5. Chỉ trả về duy nhất khối JSON hợp lệ, KHÔNG suy nghĩ kéo dài và KHÔNG có lời giải thích ngoài lề."""

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
  "skills": ["python", "fastapi", "go", "postgresql", "docker", "aws", "git", "redis", "rest api"],
  "total_years_experience": 3.5,
  "highest_degree": "bachelor",
  "certs": ["AWS Certified Developer Associate"]
}"""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Làm sạch chuỗi trả về từ LLM và chuyển đổi an toàn sang dict JSON."""
    cleaned = raw_text.strip()
    # Tìm khối nằm giữa ```json ... ``` nếu có
    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if json_block:
        cleaned = json_block.group(1).strip()
    else:
        # Nếu không có code block, tìm ký tự { đầu tiên và } cuối cùng
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def call_nvidia_llm(
    messages: List[Dict[str, str]],
    api_key: str = DEFAULT_API_KEY,
    model_name: str = DEFAULT_MODEL_NAME,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    timeout: int = 120
) -> str:
    """Hàm gửi request tới NVIDIA NIM API và nhận văn bản phản hồi hoàn chỉnh."""
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
    
    # Ưu tiên lấy trực tiếp nội dung chính
    content = message.get("content")
    if not content:
        # Nếu chạm trần token hoặc model để ở reasoning_content
        content = message.get("reasoning_content", "")
    return content


def extract_jd_entities(
    jd_text: str,
    api_key: str = DEFAULT_API_KEY,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, Any]:
    """Bóc tách thông tin Job Description (JD) thành cấu trúc chuẩn với Few-shot Prompting."""
    messages = [
        {"role": "system", "content": JD_SYSTEM_PROMPT},
        {"role": "user", "content": JD_FEW_SHOT_USER},
        {"role": "assistant", "content": JD_FEW_SHOT_ASSISTANT},
        {"role": "user", "content": f"Văn bản JD cần bóc tách:\n\"\"\"\n{jd_text}\n\"\"\""}
    ]

    raw_output = call_nvidia_llm(messages, api_key=api_key, model_name=model_name)
    parsed = clean_json_response(raw_output)

    # Đảm bảo các trường mặc định không bị thiếu
    return {
        "job_title": parsed.get("job_title", ""),
        "must_have_skills": [s.lower().strip() for s in parsed.get("must_have_skills", []) if isinstance(s, str)],
        "nice_to_have_skills": [s.lower().strip() for s in parsed.get("nice_to_have_skills", []) if isinstance(s, str)],
        "min_years_experience": float(parsed.get("min_years_experience", 0.0)),
        "min_degree": parsed.get("min_degree", "none").lower(),
        "required_certs": parsed.get("required_certs", [])
    }


def extract_cv_entities(
    cv_text: str,
    api_key: str = DEFAULT_API_KEY,
    model_name: str = DEFAULT_MODEL_NAME
) -> Dict[str, Any]:
    """Bóc tách thông tin Hồ sơ ứng viên (Resume/CV) thành cấu trúc chuẩn với Few-shot Prompting."""
    messages = [
        {"role": "system", "content": CV_SYSTEM_PROMPT},
        {"role": "user", "content": CV_FEW_SHOT_USER},
        {"role": "assistant", "content": CV_FEW_SHOT_ASSISTANT},
        {"role": "user", "content": f"Văn bản CV cần bóc tách:\n\"\"\"\n{cv_text}\n\"\"\""}
    ]

    raw_output = call_nvidia_llm(messages, api_key=api_key, model_name=model_name)
    parsed = clean_json_response(raw_output)

    # Đảm bảo các trường mặc định không bị thiếu
    return {
        "candidate_title": parsed.get("candidate_title", ""),
        "skills": [s.lower().strip() for s in parsed.get("skills", []) if isinstance(s, str)],
        "total_years_experience": float(parsed.get("total_years_experience", 0.0)),
        "highest_degree": parsed.get("highest_degree", "none").lower(),
        "certs": parsed.get("certs", [])
    }


if __name__ == "__main__":
    # Test thử nghiệm trực tiếp 1 cặp CV và JD
    sample_jd = (
        "We are looking for a Data Scientist with at least 2 years of hands-on experience. "
        "Must have strong skills in Python, PyTorch, and Machine Learning. "
        "Bachelor degree in STEM required. "
        "Preferred: Experience in NLP, Docker, and AWS."
    )

    sample_cv = (
        "Alice Smith - AI Engineer\n"
        "Experience: 3 years developing deep learning models at AI Lab.\n"
        "Tech stack: Python, PyTorch, Scikit-learn, Docker, Git.\n"
        "Education: Bachelor in Computer Science (2020).\n"
        "Certifications: Deep Learning Specialization by Coursera."
    )

    print(">>> Đang bóc tách Job Description (JD)...")
    jd_res = extract_jd_entities(sample_jd)
    print("KẾT QUẢ JD:")
    print(json.dumps(jd_res, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50 + "\n")

    print(">>> Đang bóc tách Hồ sơ ứng viên (CV)...")
    cv_res = extract_cv_entities(sample_cv)
    print("KẾT QUẢ CV:")
    print(json.dumps(cv_res, indent=2, ensure_ascii=False))


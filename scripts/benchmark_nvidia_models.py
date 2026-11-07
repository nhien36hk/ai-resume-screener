# Level 1: Thư viện chuẩn Python (Standard Library)
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import requests

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)

# Danh sách các mô hình từ 20B trở lên và SOTA trong catalog của NVIDIA
CANDIDATE_MODELS = [
    # Nhóm SOTA nổi bật được yêu cầu
    "openai/gpt-oss-20b",
    "mistralai/codestral-22b-instruct-v0.1",
    "mistralai/mixtral-8x22b-v0.1",
    "mistralai/mistral-large-4-1t-a52b",
    "google/gemma-4-31b-it",
    "google/diffusiongemma-26b-a4b-it",
    "moonshotai/kimi-k2.6",
    "01-ai/yi-large",
    "ai21labs/jamba-1.5-large-instruct",
    "databricks/dbrx-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/codellama-70b",
    "meta/llama2-70b",
    "meta/muse-glimmer-30b",
    "ibm/granite-34b-code-instruct",
    "microsoft/phi-3.5-moe-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama3-chatqa-1.5-70b",
    "nvidia/nemotron-4-340b-instruct",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "stepfun-ai/step-5-preview",
    "writer/palmyra-creative-122b",
]

TEST_CV_TEXT = """Alice Smith - AI Engineer
Summary: 3.5 years of industry experience developing Machine Learning and Deep Learning systems.
Experience:
- AI Engineer at TechLab (2021 - Present): Built NLP pipelines using Python, PyTorch, and Transformers. Deployed models on AWS using Docker.
Education:
- Bachelor in Computer Science (2017 - 2021).
Certifications:
- AWS Certified Machine Learning Specialty.
Skills: Python, PyTorch, Transformers, Docker, AWS, SQL, Git."""

GROUND_TRUTH = {
    "candidate_title": "AI Engineer",
    "total_years_experience": 3.5,
    "highest_degree": "bachelor",
    "certs": ["AWS Certified Machine Learning Specialty"],
    "skills": ["python", "pytorch", "transformers", "docker", "aws", "sql", "git"]
}

PROMPT = f"""You are a professional ATS parser. Extract information from the following CV text and output ONLY a valid JSON object.
Do NOT output any markdown code blocks, explanations, or greeting. Output strictly the JSON.

Required JSON format:
{{
  "candidate_title": string,
  "skills": [string],
  "total_years_experience": float,
  "highest_degree": "none" | "high_school" | "associate" | "bachelor" | "master" | "phd",
  "certs": [string]
}}

CV Text:
\"\"\"
{TEST_CV_TEXT}
\"\"\""""


def clean_json(text: str) -> Optional[Dict[str, Any]]:
    """Làm sạch và bóc tách khối JSON từ chuỗi kết quả."""
    if not text:
        return None
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def evaluate_extracted_json(extracted: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Đánh giá độ chính xác của JSON so với Ground Truth."""
    if not extracted or not isinstance(extracted, dict):
        return {"valid_json": False, "score": 0.0, "details": "JSON không hợp lệ hoặc rỗng"}

    score_points = 0.0
    total_points = 4.0

    # 1. candidate_title
    title = str(extracted.get("candidate_title", "")).lower()
    if "ai engineer" in title:
        score_points += 1.0

    # 2. total_years_experience (chấp nhận 3.0 hoặc 3.5)
    try:
        exp = float(extracted.get("total_years_experience", 0.0))
        if 3.0 <= exp <= 3.5:
            score_points += 1.0
        elif exp > 0:
            score_points += 0.5
    except Exception:
        pass

    # 3. highest_degree
    degree = str(extracted.get("highest_degree", "")).lower()
    if "bachelor" in degree:
        score_points += 1.0

    # 4. skills (tính độ phủ)
    skills = [str(s).lower() for s in extracted.get("skills", [])]
    matched_skills = [s for s in GROUND_TRUTH["skills"] if any(s in item for item in skills)]
    skill_acc = len(matched_skills) / len(GROUND_TRUTH["skills"])
    score_points += skill_acc

    accuracy = (score_points / total_points) * 100.0
    return {
        "valid_json": True,
        "score": round(accuracy, 1),
        "extracted": extracted
    }


def test_single_model(model_name: str, timeout: int = 20) -> Dict[str, Any]:
    """Kiểm tra gọi API 1 mô hình với timeout và đo lường latency."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    }

    t0 = time.time()
    try:
        res = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        latency = round(time.time() - t0, 2)

        if res.status_code == 200:
            data = res.json()
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning_content", "")

            parsed = clean_json(content)
            eval_res = evaluate_extracted_json(parsed)

            return {
                "model": model_name,
                "status": "ONLINE (200)",
                "latency_sec": latency,
                "valid_json": eval_res["valid_json"],
                "accuracy_pct": eval_res["score"],
                "extracted": eval_res.get("extracted", {}),
                "error": None
            }
        else:
            return {
                "model": model_name,
                "status": f"HTTP {res.status_code}",
                "latency_sec": latency,
                "valid_json": False,
                "accuracy_pct": 0.0,
                "extracted": {},
                "error": res.text[:200]
            }

    except requests.exceptions.Timeout:
        latency = round(time.time() - t0, 2)
        return {
            "model": model_name,
            "status": "TIMEOUT",
            "latency_sec": latency,
            "valid_json": False,
            "accuracy_pct": 0.0,
            "extracted": {},
            "error": f"Timeout sau {timeout}s không phản hồi"
        }
    except Exception as e:
        latency = round(time.time() - t0, 2)
        return {
            "model": model_name,
            "status": "ERROR",
            "latency_sec": latency,
            "valid_json": False,
            "accuracy_pct": 0.0,
            "extracted": {},
            "error": str(e)
        }


def run_benchmark():
    """Hàm điều phối chạy tuần tự và cập nhật kết quả liên tục."""
    print("=" * 70)
    print("BẮT ĐẦU BENCHMARK CÁC MÔ HÌNH >= 20B TRÊN NVIDIA API CATALOG")
    print(f"Tổng số mô hình cần kiểm tra: {len(CANDIDATE_MODELS)}")
    print("=" * 70 + "\n")

    results = []
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "nvidia_model_benchmark.json")

    for idx, model in enumerate(CANDIDATE_MODELS, 1):
        print(f"[{idx}/{len(CANDIDATE_MODELS)}] Đang kiểm tra: {model} ...", end=" ", flush=True)
        res = test_single_model(model, timeout=20)
        results.append(res)

        status_str = res["status"]
        lat = res["latency_sec"]
        acc = res["accuracy_pct"]

        if status_str == "ONLINE (200)" and res["valid_json"]:
            print(f"SUCCESS! -> {lat}s | Độ chính xác: {acc}%")
        elif status_str == "TIMEOUT":
            print(f"TIMEOUT ({lat}s)")
        else:
            print(f"FAILED ({status_str})")

        # Lưu checkpoint kết quả sau mỗi model
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # In bảng tổng kết
    print("\n" + "=" * 70)
    print("BẢNG TỔNG KẾT MÔ HÌNH HOẠT ĐỘNG TỐT NHẤT (ONLINE & ACCURACY > 0)")
    print("=" * 70)
    online_models = [r for r in results if r["status"] == "ONLINE (200)"]
    online_models.sort(key=lambda x: (x["valid_json"], x["accuracy_pct"], -x["latency_sec"]), reverse=True)

    print(f"{'Mô hình':<40} | {'Latency':<8} | {'Valid JSON':<10} | {'Độ chính xác':<12}")
    print("-" * 75)
    for r in online_models:
        print(f"{r['model']:<40} | {str(r['latency_sec']) + 's':<8} | {str(r['valid_json']):<10} | {str(r['accuracy_pct']) + '%':<12}")

    print(f"\nBáo cáo chi tiết đã được lưu tại: {report_path}")


if __name__ == "__main__":
    run_benchmark()


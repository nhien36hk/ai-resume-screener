from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests

from src.config import (
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_VLLM_MODEL,
    DEGREE_LEVELS,
    MAJOR_GROUPS,
)
from src.data.schemas import (
    CVEntities,
    JDEntities,
    SCHEMA_VERSION,
    get_cv_json_schema,
    get_jd_json_schema,
)
from src.features.text import number_segments

load_dotenv()

DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_API_KEY = os.getenv("NVIDIA_API_KEY", os.getenv("API_NVIDIA", ""))

JD_SYSTEM_PROMPT = f"""You are an expert AI recruiting assistant analyzing Job Descriptions for an ATS.
Your task is to extract structured entities into a valid JSON object matching the requested schema.

Standard degree levels:
{", ".join(DEGREE_LEVELS)}

Standard major groups:
{", ".join(MAJOR_GROUPS)}

Rules:
1. 'job_title': The official job title mentioned.
2. 'must_have_skills': Atomic lowercase skills explicitly stated as mandatory, required, must-have, or 'X+ years of Y'.
3. 'nice_to_have_skills': Atomic lowercase skills listed as preferred, plus, nice-to-have, or bonus.
4. 'min_years_experience': Float of minimum required industry experience. For ranges like '3-5 years', take the lower bound 3.0. If not required, return 0.0.
5. 'min_degree': Minimum required degree from standard degree levels. If not required, return 'none'.
6. 'required_certs': List of required professional certifications.
7. 'education_evidence_ids': Array of segment indices [n] containing education/degree requirements (max 3). If none, return [].
8. 'required_majors': List of required field names in lowercase (e.g. ['computer science']). If none, return [].
9. 'major_group': Single best match from standard major groups. If not specified or not required, return 'unknown'.
10. 'accepts_related_field': True if JD allows 'or related field' / 'or equivalent', otherwise false.
11. Do not hallucinate. If information is not mentioned, use default empty values."""

JD_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Numbered Job Description text:\n"
            "[1] Senior Backend Engineer\n"
            "[2] Responsibilities: Build scalable web services and cloud infrastructure.\n"
            "[3] Requirements:\n"
            "[4] 4+ years of professional backend development experience.\n"
            "[5] Proficiency in Python, Django, and PostgreSQL is mandatory.\n"
            "[6] Bachelor degree in Computer Science or related field.\n"
            "[7] Preferred: Experience with AWS, Docker, and Kubernetes.\n\n"
            "Extract entities conforming to JSON schema:"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "job_title": "Senior Backend Engineer",
            "must_have_skills": ["python", "django", "postgresql"],
            "nice_to_have_skills": ["aws", "docker", "kubernetes"],
            "min_years_experience": 4.0,
            "min_degree": "bachelor",
            "required_certs": [],
            "education_evidence_ids": [6],
            "required_majors": ["computer science"],
            "major_group": "computer_science",
            "accepts_related_field": True
        })
    },
    {
        "role": "user",
        "content": (
            "Numbered Job Description text:\n"
            "[1] Customer Service Representative\n"
            "[2] Assist incoming callers with billing questions and order tracking.\n"
            "[3] Qualifications:\n"
            "[4] Strong verbal communication and active listening skills.\n"
            "[5] High school diploma or GED required. No previous experience needed.\n\n"
            "Extract entities conforming to JSON schema:"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "job_title": "Customer Service Representative",
            "must_have_skills": ["verbal communication", "active listening"],
            "nice_to_have_skills": [],
            "min_years_experience": 0.0,
            "min_degree": "high_school",
            "required_certs": [],
            "education_evidence_ids": [5],
            "required_majors": [],
            "major_group": "unknown",
            "accepts_related_field": False
        })
    },
    {
        "role": "user",
        "content": (
            "Numbered Job Description text:\n"
            "[1] General Warehouse Helper\n"
            "[2] Move boxes, package outgoing orders, and keep warehouse organized.\n"
            "[3] Must be able to lift up to 50 lbs.\n\n"
            "Extract entities conforming to JSON schema:"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "job_title": "General Warehouse Helper",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "min_years_experience": 0.0,
            "min_degree": "none",
            "required_certs": [],
            "education_evidence_ids": [],
            "required_majors": [],
            "major_group": "unknown",
            "accepts_related_field": False
        })
    }
]

CV_SYSTEM_PROMPT = f"""You are an expert AI recruiting assistant analyzing Resumes/CVs for an ATS.
Your task is to extract structured candidate entities into a valid JSON object matching the requested schema.

Standard degree levels:
{", ".join(DEGREE_LEVELS)}

Standard major groups:
{", ".join(MAJOR_GROUPS)}

Rules:
1. 'candidate_title': The candidate's primary job title or specialization.
2. 'skills': Atomic lowercase technical and professional skills mentioned in the resume.
3. 'total_years_experience': Float of total industry experience. If total not stated, sum non-overlapping employment periods. Fresher / no experience = 0.0.
4. 'highest_degree': Highest completed educational degree level from standard degree levels (e.g. Master > Bachelor > Associate). If none, return 'none'.
5. 'certs': List of professional licenses and certifications earned.
6. 'education_evidence_ids': Array of segment indices [n] containing educational credentials and majors (max 3). If none, return [].
7. 'majors': List of declared fields of study in lowercase (e.g. ['data science', 'mathematics']). If none, return [].
8. 'major_group': Single best match from standard major groups for the highest degree. If no major specified, return 'unknown'.
9. Do not hallucinate. If information is not mentioned, use default empty values."""

CV_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Numbered Resume text:\n"
            "[1] John Doe - Senior Data Scientist\n"
            "[2] Summary: 4 years of experience building machine learning pipelines and forecasting models.\n"
            "[3] Experience: Machine Learning Engineer at QuantCorp (2020 - Present).\n"
            "[4] Education:\n"
            "[5] Master of Science in Data Science, Stanford University (2018 - 2020).\n"
            "[6] Bachelor of Science in Mathematics, UC Berkeley (2014 - 2018).\n"
            "[7] Certifications: AWS Certified Machine Learning Specialty.\n"
            "[8] Skills: Python, PyTorch, SQL, Scikit-Learn, Docker, Spark.\n\n"
            "Extract entities conforming to JSON schema:"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "candidate_title": "Senior Data Scientist",
            "skills": ["python", "pytorch", "sql", "scikit-learn", "docker", "spark"],
            "total_years_experience": 4.0,
            "highest_degree": "master",
            "certs": ["AWS Certified Machine Learning Specialty"],
            "education_evidence_ids": [5, 6],
            "majors": ["data science", "mathematics"],
            "major_group": "data_science_ai"
        })
    },
    {
        "role": "user",
        "content": (
            "Numbered Resume text:\n"
            "[1] Sarah Jenkins - Junior Web Developer\n"
            "[2] Profile: Self-taught developer with 1.5 years experience building responsive web apps.\n"
            "[3] Work History: Frontend Intern at StartHub (2023 - Present).\n"
            "[4] Education: High School Diploma, West High School (Graduated 2022).\n"
            "[5] Skills: HTML, CSS, JavaScript, React, Git.\n\n"
            "Extract entities conforming to JSON schema:"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "candidate_title": "Junior Web Developer",
            "skills": ["html", "css", "javascript", "react", "git"],
            "total_years_experience": 1.5,
            "highest_degree": "high_school",
            "certs": [],
            "education_evidence_ids": [4],
            "majors": [],
            "major_group": "unknown"
        })
    },
    {
        "role": "user",
        "content": (
            "Numbered Resume text:\n"
            "[1] Alex Rivera - Staff Accountant\n"
            "[2] Summary: CPA with 5 years experience in audit and tax preparation.\n"
            "[3] Education: Bachelor of Business Administration in Accounting, Texas A&M (2019).\n"
            "[4] Certifications: Certified Public Accountant (CPA).\n"
            "[5] Skills: Financial reporting, Tax compliance, QuickBooks, Excel, Auditing.\n\n"
            "Extract entities conforming to JSON schema:"
        )
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "candidate_title": "Staff Accountant",
            "skills": ["financial reporting", "tax compliance", "quickbooks", "excel", "auditing"],
            "total_years_experience": 5.0,
            "highest_degree": "bachelor",
            "certs": ["Certified Public Accountant (CPA)"],
            "education_evidence_ids": [3],
            "majors": ["accounting"],
            "major_group": "accounting_finance"
        })
    }
]


def build_jd_prompt(numbered_jd: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": JD_SYSTEM_PROMPT},
        *JD_FEW_SHOT,
        {"role": "user", "content": f"Numbered Job Description text:\n{numbered_jd}\n\nExtract entities conforming to JSON schema:"}
    ]


def build_cv_prompt(numbered_cv: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CV_SYSTEM_PROMPT},
        *CV_FEW_SHOT,
        {"role": "user", "content": f"Numbered Resume text:\n{numbered_cv}\n\nExtract entities conforming to JSON schema:"}
    ]


def format_evidence_text(segments: List[str], evidence_ids: List[int]) -> str:
    extracted = [segments[idx - 1] for idx in evidence_ids if 1 <= idx <= len(segments)]
    return " | ".join(extracted)


def probe_nvidia_structured_output(
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_NVIDIA_MODEL,
    api_url: str = DEFAULT_API_URL,
    timeout: int = 15,
) -> str:
    """Probes the NVIDIA API endpoint to detect supported structured output format.

    Probe results on nvidia/nemotron-3.5-super-vl-120b-a12b:
    - (a) response_format json_schema: Supported (HTTP 200). With chat_template_kwargs={"enable_thinking": False},
      it produces strict schema-guided JSON without reasoning overhead.
    - (b) nvext guided_json: Not supported (HTTP 400: unknown field `guided_json`).
    - (c) response_format json_object: Supported (HTTP 200).
    Selected mode: 'json_schema'.
    """
    key = api_key or DEFAULT_API_KEY
    if not key:
        return "json_schema"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    probe_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "code": {"type": "integer"}
        },
        "required": ["status", "code"]
    }
    messages = [{"role": "user", "content": "Respond in JSON: status='ok', code=200"}]

    # Option (a): response_format json_schema
    try:
        payload_a = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "probe_schema", "schema": probe_schema, "strict": True}
            },
            "chat_template_kwargs": {"enable_thinking": False},
            "nvext": {"max_thinking_tokens": 0}
        }
        res_a = requests.post(api_url, headers=headers, json=payload_a, timeout=timeout)
        if res_a.status_code == 200:
            content = res_a.json()["choices"][0]["message"].get("content")
            if content:
                return "json_schema"
    except Exception:
        pass

    # Option (b): nvext guided_json
    try:
        payload_b = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0,
            "nvext": {"guided_json": probe_schema}
        }
        res_b = requests.post(api_url, headers=headers, json=payload_b, timeout=timeout)
        if res_b.status_code == 200:
            return "nvext"
    except Exception:
        pass

    # Option (c): response_format json_object
    try:
        payload_c = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}
        }
        res_c = requests.post(api_url, headers=headers, json=payload_c, timeout=timeout)
        if res_c.status_code == 200:
            return "json_object"
    except Exception:
        pass

    return "json_schema"


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / max(rpm, 1.0)
        self.lock = threading.Lock()
        self.last_called = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_called
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_called = time.time()


class BaseExtractorBackend(ABC):
    @abstractmethod
    def extract_one(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str,
        max_tokens: int = 1024,
    ) -> Tuple[Dict[str, Any], int, int]:
        """Returns (parsed_json_dict, retry_count, is_truncated_flag)."""
        pass


class NvidiaAPIBackend(BaseExtractorBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_NVIDIA_MODEL,
        api_url: str = DEFAULT_API_URL,
        rpm: float = 35.0,
        timeout: int = 90,
        max_retries: int = 4,
    ):
        self.api_key = api_key or DEFAULT_API_KEY
        if not self.api_key:
            raise ValueError("Chưa cấu hình API Key. Thêm NVIDIA_API_KEY hoặc API_NVIDIA vào .env")
        self.model_name = model_name
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rpm)
        self.format_mode = probe_nvidia_structured_output(self.api_key, self.model_name, self.api_url)

    def _send_request(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str,
        max_tokens: int,
    ) -> Tuple[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "nvext": {"max_thinking_tokens": 0},
        }

        if self.format_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True}
            }
        elif self.format_mode == "nvext":
            payload["nvext"]["guided_json"] = schema
        else:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            self.rate_limiter.wait()
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    content = choice["message"].get("content")
                    finish_reason = choice.get("finish_reason", "stop")
                    if content is None or not content.strip():
                        raise RuntimeError(f"NVIDIA API trả về content rỗng (finish_reason: {finish_reason})")
                    return content.strip(), finish_reason

                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    sleep_sec = float(retry_after) if retry_after else (2 ** attempt * 2.0)
                    time.sleep(sleep_sec)
                    continue

                raise RuntimeError(f"NVIDIA API Error HTTP {resp.status_code}: {resp.text}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt * 2.0)

        raise RuntimeError(f"Vượt quá số lần retry ({self.max_retries}) với NVIDIA API")

    def extract_one(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str,
        max_tokens: int = 1024,
    ) -> Tuple[Dict[str, Any], int, int]:
        retries = 0
        truncated = 0
        current_max_tokens = max_tokens
        curr_messages = list(messages)

        for turn in range(3):
            content, finish_reason = self._send_request(
                curr_messages, schema, schema_name, current_max_tokens
            )

            if finish_reason == "length":
                truncated = 1
                retries += 1
                current_max_tokens *= 2
                continue

            try:
                parsed = json.loads(content)
                return parsed, retries, truncated
            except json.JSONDecodeError as err:
                retries += 1
                if turn == 2:
                    raise RuntimeError(f"Model không sinh JSON hợp lệ sau {retries} lần retry: {err}")
                curr_messages.append({"role": "assistant", "content": content})
                curr_messages.append({
                    "role": "user",
                    "content": f"The response was not valid JSON ({err}). Output ONLY valid JSON adhering to the schema."
                })

        raise RuntimeError("Không thể trích xuất entity hợp lệ qua API")


class VLLMBackend(BaseExtractorBackend):
    def __init__(
        self,
        model_name: str = DEFAULT_VLLM_MODEL,
        gpu_memory_utilization: float = 0.88,
        max_model_len: int = 8192,
    ):
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        try:
            from vllm import LLM
            from vllm.sampling_params import SamplingParams
            try:
                from vllm.sampling_params import GuidedDecodingParams
                self.guided_arg = "guided_decoding"
            except ImportError:
                from vllm.sampling_params import StructuredOutputsParams as GuidedDecodingParams
                self.guided_arg = "structured_outputs"
        except ImportError as e:
            raise ImportError(f"Chưa cài đặt vLLM hoặc môi trường không hỗ trợ: {e}")

        self.GuidedDecodingParams = GuidedDecodingParams
        self.SamplingParams = SamplingParams
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype="fp8",
            enable_prefix_caching=True,
            seed=42,
        )

    def extract_one(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        schema_name: str,
        max_tokens: int = 1024,
    ) -> Tuple[Dict[str, Any], int, int]:
        batch_res = self.extract_batch([messages], schema, max_tokens=max_tokens)
        return batch_res[0]

    def extract_batch(
        self,
        prompts: List[List[Dict[str, str]]],
        schema: Dict[str, Any],
        max_tokens: int = 1024,
    ) -> List[Tuple[Dict[str, Any], int, int]]:
        params = self.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            **{self.guided_arg: self.GuidedDecodingParams(json=schema)}
        )
        outputs = self.llm.chat(
            messages=prompts,
            sampling_params=params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": False}
        )

        results = []
        for out in outputs:
            raw_text = out.outputs[0].text
            finish_reason = out.outputs[0].finish_reason
            truncated = 1 if finish_reason == "length" else 0
            try:
                data = json.loads(raw_text)
                results.append((data, 0, truncated))
            except json.JSONDecodeError as err:
                raise RuntimeError(f"vLLM output không parse được JSON: {err} | Raw: {raw_text[:200]}")
        return results


def extract_jd_entities(
    jd_text: str,
    backend: Optional[BaseExtractorBackend] = None,
    model_name: str = DEFAULT_NVIDIA_MODEL,
) -> Dict[str, Any]:
    if backend is None:
        backend = NvidiaAPIBackend(model_name=model_name)

    numbered_text, segments = number_segments(jd_text)
    messages = build_jd_prompt(numbered_text)
    schema = get_jd_json_schema()

    raw_json, _, _ = backend.extract_one(messages, schema, "jd_schema")
    validated = JDEntities.model_validate(raw_json)

    entity_dict = validated.model_dump()
    entity_dict["major_evidence"] = format_evidence_text(segments, validated.education_evidence_ids)
    entity_dict["_schema_version"] = SCHEMA_VERSION
    entity_dict["_model"] = getattr(backend, "model_name", model_name)
    entity_dict["_backend"] = "nvidia" if isinstance(backend, NvidiaAPIBackend) else "vllm"
    entity_dict["_extracted_at"] = datetime.now(timezone.utc).isoformat()
    return entity_dict


def extract_cv_entities(
    cv_text: str,
    backend: Optional[BaseExtractorBackend] = None,
    model_name: str = DEFAULT_NVIDIA_MODEL,
) -> Dict[str, Any]:
    if backend is None:
        backend = NvidiaAPIBackend(model_name=model_name)

    numbered_text, segments = number_segments(cv_text)
    messages = build_cv_prompt(numbered_text)
    schema = get_cv_json_schema()

    raw_json, _, _ = backend.extract_one(messages, schema, "cv_schema")
    validated = CVEntities.model_validate(raw_json)

    entity_dict = validated.model_dump()
    entity_dict["major_evidence"] = format_evidence_text(segments, validated.education_evidence_ids)
    entity_dict["_schema_version"] = SCHEMA_VERSION
    entity_dict["_model"] = getattr(backend, "model_name", model_name)
    entity_dict["_backend"] = "nvidia" if isinstance(backend, NvidiaAPIBackend) else "vllm"
    entity_dict["_extracted_at"] = datetime.now(timezone.utc).isoformat()
    return entity_dict

"""LLM entity extraction for JDs and CVs, through the NVIDIA API or a local vLLM engine."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import threading
import time

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import requests

from src.config import DEFAULT_NVIDIA_MODEL, DEFAULT_VLLM_MODEL, DEGREE_LEVELS, MAJOR_GROUPS
from src.data.schemas import SCHEMA_VERSION, CVEntities, JDEntities, json_schema
from src.features.text import number_segments

load_dotenv()

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
RETRY_STATUS = {429, 500, 502, 503, 504}

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
5. 'required_certs': List of required professional certifications.
6. 'education_evidence_ids': Array of segment indices [n] containing education/degree requirements (max 3). If none, return [].
7. 'min_degree': Minimum required degree from standard degree levels. If not required, return 'none'.
8. 'required_majors': List of required field names in lowercase (e.g. ['computer science']). If none, return [].
9. 'major_group': Single best match from standard major groups. If not specified or not required, return 'unknown'.
10. 'accepts_related_field': True if JD allows 'or related field' / 'or equivalent', otherwise false.
11. Do not hallucinate. If information is not mentioned, use default empty values."""

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
4. 'certs': List of professional licenses and certifications earned.
5. 'education_evidence_ids': Array of segment indices [n] containing educational credentials and majors (max 3). If none, return [].
6. 'highest_degree': Highest completed educational degree level from standard degree levels (e.g. Master > Bachelor > Associate). If none, return 'none'.
7. 'majors': List of declared fields of study in lowercase (e.g. ['data science', 'mathematics']). If none, return [].
8. 'major_group': Single best match from standard major groups for the highest degree. If no major specified, return 'unknown'.
9. Do not hallucinate. If information is not mentioned, use default empty values."""

JD_LABEL = "Numbered Job Description text"
CV_LABEL = "Numbered Resume text"


def _user_msg(label: str, numbered_text: str) -> dict:
    return {"role": "user", "content": f"{label}:\n{numbered_text}\n\nExtract entities conforming to JSON schema:"}


def _shot(label: str, lines: list[str], answer: dict) -> list[dict]:
    numbered = "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, 1))
    return [_user_msg(label, numbered), {"role": "assistant", "content": json.dumps(answer)}]


JD_FEW_SHOT = [
    *_shot(JD_LABEL, [
        "Senior Backend Engineer",
        "Responsibilities: Build scalable web services and cloud infrastructure.",
        "Requirements:",
        "4+ years of professional backend development experience.",
        "Proficiency in Python, Django, and PostgreSQL is mandatory.",
        "Bachelor degree in Computer Science or related field.",
        "Preferred: Experience with AWS, Docker, and Kubernetes.",
    ], {
        "job_title": "Senior Backend Engineer",
        "must_have_skills": ["python", "django", "postgresql"],
        "nice_to_have_skills": ["aws", "docker", "kubernetes"],
        "min_years_experience": 4.0,
        "required_certs": [],
        "education_evidence_ids": [6],
        "min_degree": "bachelor",
        "required_majors": ["computer science"],
        "major_group": "computer_science",
        "accepts_related_field": True,
    }),
    *_shot(JD_LABEL, [
        "Customer Service Representative",
        "Assist incoming callers with billing questions and order tracking.",
        "Qualifications:",
        "Strong verbal communication and active listening skills.",
        "High school diploma or GED required. No previous experience needed.",
    ], {
        "job_title": "Customer Service Representative",
        "must_have_skills": ["verbal communication", "active listening"],
        "nice_to_have_skills": [],
        "min_years_experience": 0.0,
        "required_certs": [],
        "education_evidence_ids": [5],
        "min_degree": "high_school",
        "required_majors": [],
        "major_group": "unknown",
        "accepts_related_field": False,
    }),
    *_shot(JD_LABEL, [
        "General Warehouse Helper",
        "Move boxes, package outgoing orders, and keep warehouse organized.",
        "Must be able to lift up to 50 lbs.",
    ], {
        "job_title": "General Warehouse Helper",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "min_years_experience": 0.0,
        "required_certs": [],
        "education_evidence_ids": [],
        "min_degree": "none",
        "required_majors": [],
        "major_group": "unknown",
        "accepts_related_field": False,
    }),
]

CV_FEW_SHOT = [
    *_shot(CV_LABEL, [
        "John Doe - Senior Data Scientist",
        "Summary: 4 years of experience building machine learning pipelines and forecasting models.",
        "Experience: Machine Learning Engineer at QuantCorp (2020 - Present).",
        "Education:",
        "Master of Science in Data Science, Stanford University (2018 - 2020).",
        "Bachelor of Science in Mathematics, UC Berkeley (2014 - 2018).",
        "Certifications: AWS Certified Machine Learning Specialty.",
        "Skills: Python, PyTorch, SQL, Scikit-Learn, Docker, Spark.",
    ], {
        "candidate_title": "Senior Data Scientist",
        "skills": ["python", "pytorch", "sql", "scikit-learn", "docker", "spark"],
        "total_years_experience": 4.0,
        "certs": ["aws certified machine learning specialty"],
        "education_evidence_ids": [5, 6],
        "highest_degree": "master",
        "majors": ["data science", "mathematics"],
        "major_group": "data_science_ai",
    }),
    *_shot(CV_LABEL, [
        "Sarah Jenkins - Junior Web Developer",
        "Profile: Self-taught developer with 1.5 years experience building responsive web apps.",
        "Work History: Frontend Intern at StartHub (2023 - Present).",
        "Education: High School Diploma, West High School (Graduated 2022).",
        "Skills: HTML, CSS, JavaScript, React, Git.",
    ], {
        "candidate_title": "Junior Web Developer",
        "skills": ["html", "css", "javascript", "react", "git"],
        "total_years_experience": 1.5,
        "certs": [],
        "education_evidence_ids": [4],
        "highest_degree": "high_school",
        "majors": [],
        "major_group": "unknown",
    }),
    *_shot(CV_LABEL, [
        "Alex Rivera - Staff Accountant",
        "Summary: CPA with 5 years experience in audit and tax preparation.",
        "Education: Bachelor of Business Administration in Accounting, Texas A&M (2019).",
        "Certifications: Certified Public Accountant (CPA).",
        "Skills: Financial reporting, Tax compliance, QuickBooks, Excel, Auditing.",
    ], {
        "candidate_title": "Staff Accountant",
        "skills": ["financial reporting", "tax compliance", "quickbooks", "excel", "auditing"],
        "total_years_experience": 5.0,
        "certs": ["certified public accountant (cpa)"],
        "education_evidence_ids": [3],
        "highest_degree": "bachelor",
        "majors": ["accounting"],
        "major_group": "accounting_finance",
    }),
]

# kind -> (schema, fixed prompt prefix, label of the user message). The prefix is identical for every
# document of a kind, so vLLM's prefix cache computes it only once.
TASKS = {
    "jd": (JDEntities, [{"role": "system", "content": JD_SYSTEM_PROMPT}, *JD_FEW_SHOT], JD_LABEL),
    "cv": (CVEntities, [{"role": "system", "content": CV_SYSTEM_PROMPT}, *CV_FEW_SHOT], CV_LABEL),
}


class NvidiaAPIBackend:
    """Calls the hosted API in parallel threads, under a requests-per-minute limit.

    Probed on nvidia/nemotron-3.5-super-vl-120b-a12b: response_format json_schema works with
    enable_thinking=False; nvext.guided_json is rejected (HTTP 400).
    """

    name = "nvidia"

    def __init__(self, model_name: str = DEFAULT_NVIDIA_MODEL, workers: int = 4, rpm: float = 35.0,
                 timeout: int = 90, max_retries: int = 4):
        self.api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("API_NVIDIA")
        if not self.api_key:
            raise ValueError("Chưa có API key. Thêm NVIDIA_API_KEY hoặc API_NVIDIA vào .env")
        self.model_name = model_name
        self.workers = workers
        self.timeout = timeout
        self.max_retries = max_retries
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _wait_rate_limit(self) -> None:
        with self._lock:
            delay = self._last_call + self._interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last_call = time.monotonic()

    def _post(self, messages: list[dict], schema: dict, max_tokens: int) -> tuple[str, str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema["title"], "schema": schema, "strict": True},
            },
            "chat_template_kwargs": {"enable_thinking": False},
            "nvext": {"max_thinking_tokens": 0},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        for attempt in range(self.max_retries):
            self._wait_rate_limit()
            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2.0 * 2 ** attempt)
                continue
            if resp.status_code == 200:
                choice = resp.json()["choices"][0]
                return (choice["message"].get("content") or "").strip(), choice.get("finish_reason") or "stop"
            if resp.status_code not in RETRY_STATUS:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(_retry_after(resp, attempt))
        raise RuntimeError(f"Hết {self.max_retries} lần retry API")

    def generate(self, prompts: list[list[dict]], schema: dict, max_tokens: int) -> list[tuple[str, str]]:
        def one(messages):
            try:
                return self._post(messages, schema, max_tokens)
            except Exception as e:
                return "", f"error: {e}"

        pool = ThreadPoolExecutor(self.workers)
        try:
            return list(pool.map(one, prompts))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # on Ctrl+C, drop queued requests instead of waiting for them


def _retry_after(resp: requests.Response, attempt: int) -> float:
    try:
        return float(resp.headers["Retry-After"])
    except (KeyError, ValueError):  # missing, or an HTTP date
        return 2.0 * 2 ** attempt


class VLLMBackend:
    """Runs a local model with vLLM. One generate() call is one batched llm.chat over all prompts."""

    name = "vllm"

    def __init__(self, model_name: str = DEFAULT_VLLM_MODEL, gpu_memory_utilization: float = 0.88,
                 max_model_len: int = 8192):
        import multiprocessing
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        from vllm import LLM, SamplingParams  # lazy: machines without vLLM can still use the API backend
        try:
            from vllm.sampling_params import GuidedDecodingParams as Guided
            self._guided_arg = "guided_decoding"
        except ImportError:
            from vllm.sampling_params import StructuredOutputsParams as Guided
            self._guided_arg = "structured_outputs"

        self._Guided = Guided
        self._SamplingParams = SamplingParams
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype="fp8",
            enable_prefix_caching=True,
            seed=42,
        )

    def generate(self, prompts: list[list[dict]], schema: dict, max_tokens: int) -> list[tuple[str, str]]:
        params = self._SamplingParams(
            temperature=0.0, max_tokens=max_tokens, **{self._guided_arg: self._Guided(json=schema)}
        )
        outputs = self.llm.chat(
            prompts, sampling_params=params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False}
        )
        return [(out.outputs[0].text, out.outputs[0].finish_reason or "stop") for out in outputs]


def format_evidence_text(segments: list[str], evidence_ids: list[int]) -> str:
    return " | ".join(segments[i - 1] for i in evidence_ids if 1 <= i <= len(segments))


def _parse(model_cls: type[BaseModel], text: str, finish_reason: str) -> tuple[BaseModel | None, str | None]:
    if finish_reason == "length":
        return None, "truncated"
    if finish_reason.startswith("error"):
        return None, finish_reason
    try:
        return model_cls.model_validate_json(text), None
    except ValidationError as e:  # covers broken JSON too
        return None, f"invalid: {str(e)[:500]}"


def _retry_prompt(prompt: list[dict], text: str, error: str) -> list[dict]:
    if not error.startswith("invalid"):
        return prompt
    return prompt + [
        {"role": "assistant", "content": text},
        {"role": "user", "content": f"That JSON failed validation: {error}\nReturn only the corrected JSON."},
    ]


def extract_many(kind: str, texts: list[str], backend, max_tokens: int = 1024) -> list[dict | str]:
    """Extract entities for every text in one backend batch. Returns an entity dict, or an error message, per text.

    Failed items (truncated, invalid, API error) are retried once together, with twice the token budget.
    """
    model_cls, prefix, label = TASKS[kind]
    schema = json_schema(model_cls)
    numbered = [number_segments(t) for t in texts]
    prompts = [prefix + [_user_msg(label, text)] for text, _ in numbered]

    outputs = backend.generate(prompts, schema, max_tokens)
    parsed = [_parse(model_cls, *out) for out in outputs]
    bad = [i for i, (entity, _) in enumerate(parsed) if entity is None]
    if bad:
        retry_prompts = [_retry_prompt(prompts[i], outputs[i][0], parsed[i][1]) for i in bad]
        for i, out in zip(bad, backend.generate(retry_prompts, schema, max_tokens * 2)):
            parsed[i] = _parse(model_cls, *out)

    meta = {
        "_schema_version": SCHEMA_VERSION,
        "_model": backend.model_name,
        "_backend": backend.name,
        "_extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    return [
        {**entity.model_dump(), "major_evidence": format_evidence_text(segments, entity.education_evidence_ids), **meta}
        if entity else error
        for (entity, error), (_, segments) in zip(parsed, numbered)
    ]

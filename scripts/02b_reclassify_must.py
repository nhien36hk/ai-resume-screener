"""Phân loại lại must-skills của JD: chọn tối đa 5 kỹ năng bắt buộc thật sự từ danh sách đã trích (must ∪ nice).

Chỉ đụng các JD có > 5 must. Must cũ giữ ở `must_have_skills_orig` (dòng "spec gốc" của E4).
JD nào LLM trả lỗi thì KHÔNG ghi gì -> chạy lại script là thử lại đúng các JD đó.
    python scripts/02b_reclassify_must.py --backend vllm
    python scripts/02b_reclassify_must.py --backend nvidia
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_NVIDIA_MODEL, DEFAULT_VLLM_MODEL, JD_CACHE_PATH, PROCESSED_DIR
from src.utils import load_json, save_json

MAX_MUST = 5
SYSTEM_PROMPT = (
    "You read a Job Description and a list of skills already extracted from it. "
    f"Select AT MOST {MAX_MUST} skills that the JD states as absolute, non-negotiable requirements "
    "(required / must have / minimum qualifications). Copy each skill string exactly as it appears in the list. "
    "Preferred, nice-to-have or merely mentioned skills must NOT be selected."
)
# Gốc là object (như mọi schema trong extractor.py): API strict json_schema không nhận gốc là mảng.
SCHEMA = {
    "title": "MustSkills",
    "type": "object",
    "properties": {"must_have_skills": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_MUST}},
    "required": ["must_have_skills"],
    "additionalProperties": False,
}


def build_prompt(jd_text: str, pool: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Job Description:\n{jd_text}\n\nExtracted skills: {json.dumps(pool)}"},
    ]


def parse_must(text: str, pool: list[str]) -> list[str] | None:
    """Tập con (≤ 5) của pool theo output LLM; None nếu output hỏng -> không ghi, lần sau thử lại."""
    try:
        picked = json.loads(text)["must_have_skills"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(picked, list):
        return None
    picked = [s.strip().lower() for s in picked if isinstance(s, str)]
    must = [s for s in dict.fromkeys(picked) if s in pool][:MAX_MUST]
    return None if picked and not must else must  # trả về toàn chuỗi lạ (không có trong pool) = hỏng


def skill_pool(e: dict) -> list[str]:
    """Danh sách gửi LLM và dùng để đối chiếu kết quả: phải là một."""
    return list(dict.fromkeys(e["must_have_skills"] + e.get("nice_to_have_skills", [])))


def plan(cache: dict) -> list[str]:
    """JD cần phân loại: chưa xử lý và có > MAX_MUST must."""
    return [j for j, e in cache.items()
            if "must_have_skills_orig" not in e and len(e.get("must_have_skills", [])) > MAX_MUST]


def apply_results(cache: dict, todo: list[str], outputs: list[tuple[str, str]]) -> list[str]:
    """Ghi kết quả vào cache; trả danh sách JD lỗi (không bị ghi)."""
    failed = []
    for j_id, (text, _finish) in zip(todo, outputs):
        e = cache[j_id]
        pool = skill_pool(e)
        must = parse_must(text, pool)
        if must is None:
            failed.append(j_id)
            continue
        e["must_have_skills_orig"] = list(e["must_have_skills"])
        e["must_have_skills"] = must
        e["nice_to_have_skills"] = [s for s in pool if s not in must]
    for e in cache.values():  # JD vốn ≤ 5 must: giữ nguyên, chỉ ghi lại must gốc cho E4
        if len(e.get("must_have_skills", [])) <= MAX_MUST:
            e.setdefault("must_have_skills_orig", list(e["must_have_skills"]))
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["nvidia", "vllm"], default="vllm")
    p.add_argument("--model", help="Mặc định theo backend.")
    p.add_argument("--cache-path", type=Path, default=JD_CACHE_PATH)
    args = p.parse_args()

    cache = load_json(args.cache_path, {})
    todo = plan(cache)
    print(f"{len(todo)}/{len(cache)} JD cần phân loại lại (must > {MAX_MUST}).")
    if not todo:
        return

    backup = args.cache_path.with_name(args.cache_path.stem + ".pre_reclassify.json")
    if not backup.exists():
        shutil.copy2(args.cache_path, backup)

    jd_texts = {}
    for split in ("train", "val", "test"):
        df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet", columns=["job_id", "job_description_text"])
        jd_texts.update(zip(df["job_id"].astype(str), df["job_description_text"]))
    missing = [j for j in todo if j not in jd_texts]
    if missing:
        sys.exit(f"Không tìm thấy văn bản JD cho {len(missing)} job_id, vd {missing[:3]} — dừng, chưa ghi gì.")

    from src.data.extractor import NvidiaAPIBackend, VLLMBackend  # lazy: vllm chỉ có trên máy GPU
    backend = (VLLMBackend(args.model or DEFAULT_VLLM_MODEL) if args.backend == "vllm"
               else NvidiaAPIBackend(args.model or DEFAULT_NVIDIA_MODEL))

    prompts = [build_prompt(jd_texts[j], skill_pool(cache[j])) for j in todo]
    failed = apply_results(cache, todo, backend.generate(prompts, SCHEMA, max_tokens=256))
    save_json(args.cache_path, cache)

    print(f"Đã phân loại lại {len(todo) - len(failed)}/{len(todo)} JD. Backup: {backup}")
    if failed:
        print(f"LỖI {len(failed)} JD (chưa ghi, chạy lại để thử tiếp): {failed[:10]}")
        sys.exit(1)


if __name__ == "__main__":
    main()

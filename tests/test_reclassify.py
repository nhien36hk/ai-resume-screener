import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("reclassify", Path(__file__).resolve().parent.parent / "scripts" / "02b_reclassify_must.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

POOL = ["python", "sql", "docker", "aws", "git", "linux", "jira"]


def test_parse_must_keeps_only_pool_subset():
    assert rc.parse_must('{"must_have_skills": ["Python", "SQL", "kubernetes"]}', POOL) == ["python", "sql"]
    assert rc.parse_must('{"must_have_skills": []}', POOL) == []           # LLM nói không có must: hợp lệ
    assert rc.parse_must("", POOL) is None                                # backend lỗi
    assert rc.parse_must('{"must_have_skills": ["rust"]}', POOL) is None  # toàn chuỗi lạ = hỏng
    assert len(rc.parse_must('{"must_have_skills": ' + str(POOL).replace("'", '"') + "}", POOL)) == 5


def test_failed_output_is_not_written_and_retried():
    cache = {
        "JD_OK": {"must_have_skills": POOL[:6], "nice_to_have_skills": ["jira"]},
        "JD_ERR": {"must_have_skills": POOL[:6], "nice_to_have_skills": []},
        "JD_SMALL": {"must_have_skills": ["python"], "nice_to_have_skills": []},
    }
    todo = rc.plan(cache)
    assert todo == ["JD_OK", "JD_ERR"]

    outputs = [('{"must_have_skills": ["python", "docker"]}', "stop"), ("", "error: HTTP 400")]
    failed = rc.apply_results(cache, todo, outputs)

    assert failed == ["JD_ERR"]
    assert cache["JD_OK"]["must_have_skills"] == ["python", "docker"]
    assert cache["JD_OK"]["must_have_skills_orig"] == POOL[:6]
    assert set(cache["JD_OK"]["nice_to_have_skills"]) == set(POOL) - {"python", "docker"}
    assert "must_have_skills_orig" not in cache["JD_ERR"]          # không đánh dấu -> lần sau thử lại
    assert cache["JD_SMALL"]["must_have_skills_orig"] == ["python"]
    assert rc.plan(cache) == ["JD_ERR"]


def test_big_pool_still_valid_after_reclassify():
    # pool 45 (lớn nhất trên cache), LLM chọn 1 -> nice 44: vẫn phải qua schema, không thì 03_merge bỏ JD
    from src.data.schemas import SCHEMA_VERSION, JDEntities, is_current
    skills = [f"skill{i}" for i in range(45)]
    jd = {"_schema_version": SCHEMA_VERSION, "job_title": "x", "must_have_skills": skills[:40],
          "nice_to_have_skills": skills[40:], "min_years_experience": 0, "required_certs": [],
          "education_evidence_ids": [], "min_degree": "none", "required_majors": [],
          "major_group": "unknown", "accepts_related_field": False}
    cache = {"JD": jd}
    assert rc.apply_results(cache, rc.plan(cache), [('{"must_have_skills": ["skill0"]}', "stop")]) == []
    assert len(jd["nice_to_have_skills"]) == 44
    assert is_current(JDEntities, jd)

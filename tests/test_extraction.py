import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from src.config import DEGREE_LEVELS, MAJOR_GROUPS
from src.data.extractor import (
    BaseExtractorBackend,
    extract_cv_entities,
    extract_jd_entities,
    format_evidence_text,
)
from src.data.schemas import (
    CVEntities,
    JDEntities,
    SCHEMA_VERSION,
    get_cv_json_schema,
    get_jd_json_schema,
)
from src.utils import load_json, save_json


def test_schema_normalization_and_validation():
    # 1. Normalization: lowercase, strip, dedup with order preserved
    jd_raw = {
        "job_title": "Backend Lead",
        "must_have_skills": [" Python ", "python", "Django", "DJANGO"],
        "nice_to_have_skills": [" AWS ", "aws", "Docker"],
        "min_years_experience": "3-5 years",
        "min_degree": "Bachelor ",
        "required_certs": [" AWS Certified Architect ", "aws certified architect"],
        "education_evidence_ids": [1, 2],
        "required_majors": [" Computer Science ", "computer science"],
        "major_group": "computer_science",
        "accepts_related_field": True,
    }
    jd = JDEntities.model_validate(jd_raw)
    assert jd.must_have_skills == ["python", "django"]
    assert jd.nice_to_have_skills == ["aws", "docker"]
    assert jd.min_years_experience == 3.0
    assert jd.min_degree == "bachelor"
    assert jd.required_certs == ["AWS Certified Architect"]
    assert jd.required_majors == ["computer science"]

    # 2. Invalid degree enum value must raise ValidationError
    with pytest.raises(ValidationError):
        JDEntities.model_validate({**jd_raw, "min_degree": "invalid_degree"})

    # 3. Invalid major_group enum value must raise ValidationError
    with pytest.raises(ValidationError):
        JDEntities.model_validate({**jd_raw, "major_group": "non_existent_major"})

    # 4. maxItems enforcement (skills <= 40)
    oversized_skills = [f"skill_{i}" for i in range(45)]
    with pytest.raises(ValidationError):
        JDEntities.model_validate({**jd_raw, "must_have_skills": oversized_skills})

    # 5. CV schema normalization
    cv_raw = {
        "candidate_title": "ML Engineer",
        "skills": [" PyTorch ", "pytorch", "SQL"],
        "total_years_experience": "4.5",
        "highest_degree": "Master",
        "certs": ["Deep Learning Specialization"],
        "education_evidence_ids": [2],
        "majors": [" Data Science ", "data science"],
        "major_group": "data_science_ai",
    }
    cv = CVEntities.model_validate(cv_raw)
    assert cv.skills == ["pytorch", "sql"]
    assert cv.total_years_experience == 4.5
    assert cv.highest_degree == "master"
    assert cv.majors == ["data science"]


def test_needs_extraction():
    import importlib
    extract_mod = importlib.import_module("scripts.02_extract_entities")
    needs_extraction = extract_mod.needs_extraction

    # None or empty entry needs extraction
    assert needs_extraction(None, JDEntities) is True
    assert needs_extraction({}, JDEntities) is True

    # Old entry without major_group or with _schema_version < 2 needs extraction
    old_jd = {
        "job_title": "Engineer",
        "must_have_skills": ["python"],
        "min_years_experience": 2.0,
        "min_degree": "bachelor",
        "required_certs": [],
    }
    assert needs_extraction(old_jd, JDEntities) is True

    # Complete valid entry with current SCHEMA_VERSION does NOT need extraction
    valid_jd = {
        "job_title": "Engineer",
        "must_have_skills": ["python"],
        "nice_to_have_skills": [],
        "min_years_experience": 2.0,
        "min_degree": "bachelor",
        "required_certs": [],
        "education_evidence_ids": [1],
        "required_majors": ["computer science"],
        "major_group": "computer_science",
        "accepts_related_field": True,
        "_schema_version": SCHEMA_VERSION,
    }
    assert needs_extraction(valid_jd, JDEntities) is False


def test_parser_and_evidence_ids():
    segments = ["Header", "Required: Bachelor in Computer Science", "Skills: Python, SQL"]

    # In-bounds evidence_ids format correctly
    evidence_text = format_evidence_text(segments, [2])
    assert evidence_text == "Required: Bachelor in Computer Science"

    # Out of bounds evidence_ids are safely ignored
    evidence_text_oob = format_evidence_text(segments, [2, 99, -1, 0])
    assert evidence_text_oob == "Required: Bachelor in Computer Science"

    # Extraction fails when backend returns invalid JSON - never returns default fake entity
    class BrokenMockBackend(BaseExtractorBackend):
        def extract_one(self, messages, schema, schema_name, max_tokens=1024):
            raise RuntimeError("Model generation failed")

    broken_backend = BrokenMockBackend()
    with pytest.raises(RuntimeError):
        extract_jd_entities("Job description text", backend=broken_backend)


def test_atomic_save_and_strict_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "cache.json"

        # 1. Atomic write succeeds and leaves no temporary files
        data = {"key": "value", "numbers": [1, 2, 3]}
        save_json(tmp_path, data)
        assert tmp_path.exists()
        assert list(Path(tmp_dir).glob("*.tmp*")) == []

        loaded = load_json(tmp_path)
        assert loaded == data

        # 2. Non-existent file returns default value
        missing_path = Path(tmp_dir) / "missing.json"
        assert load_json(missing_path, default_val={"default": True}) == {"default": True}

        # 3. Corrupted file raises JSONDecodeError, never returns empty dict silently
        corrupted_path = Path(tmp_dir) / "corrupt.json"
        with open(corrupted_path, "w", encoding="utf-8") as f:
            f.write("{invalid json...")

        with pytest.raises(json.JSONDecodeError):
            load_json(corrupted_path)


class MockCoordinatorBackend(BaseExtractorBackend):
    def __init__(self):
        self.model_name = "mock-model"

    def extract_one(self, messages, schema, schema_name, max_tokens=1024):
        if schema_name == "jd_schema":
            return {
                "job_title": "Software Engineer",
                "must_have_skills": ["python", "sql"],
                "nice_to_have_skills": ["docker"],
                "min_years_experience": 2.0,
                "min_degree": "bachelor",
                "required_certs": [],
                "education_evidence_ids": [1],
                "required_majors": ["computer science"],
                "major_group": "computer_science",
                "accepts_related_field": True,
            }, 0, 0
        else:
            return {
                "candidate_title": "Software Engineer",
                "skills": ["python", "sql"],
                "total_years_experience": 3.0,
                "highest_degree": "bachelor",
                "certs": [],
                "education_evidence_ids": [1],
                "majors": ["computer science"],
                "major_group": "computer_science",
            }, 0, 0


def test_mock_backend_coordination_end_to_end():
    mock_backend = MockCoordinatorBackend()
    sample_cvs = [
        "Candidate 1 with 3 years Python development experience. Bachelor in CS.",
        "Candidate 2 with 2 years Python and SQL experience. Bachelor in CS.",
        "Candidate 3 with 5 years experience in Software Engineering.",
    ]

    extracted_results = []
    for cv_text in sample_cvs:
        ent = extract_cv_entities(cv_text, backend=mock_backend)
        extracted_results.append(ent)

    assert len(extracted_results) == 3
    for ent in extracted_results:
        assert ent["_schema_version"] == SCHEMA_VERSION
        assert ent["_backend"] == "vllm" or ent["_backend"] == "nvidia"
        assert ent["highest_degree"] == "bachelor"
        assert ent["major_group"] == "computer_science"
        assert "major_evidence" in ent

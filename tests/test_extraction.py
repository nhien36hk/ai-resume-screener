import importlib.util
import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
import pytest

from src.data.extractor import extract_many, format_evidence_text
from src.data.schemas import SCHEMA_VERSION, CVEntities, JDEntities, is_current, json_schema
from src.features.text import segment_text
from src.utils import compute_text_hash, load_json, save_json

ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CV_OK = {
    "candidate_title": "Engineer",
    "skills": ["Python", "python", " SQL "],
    "total_years_experience": 3,
    "certs": [],
    "education_evidence_ids": [1],
    "highest_degree": "bachelor",
    "majors": ["computer science"],
    "major_group": "computer_science",
}
JD_OK = {
    "job_title": "Engineer",
    "must_have_skills": ["python"],
    "nice_to_have_skills": [],
    "min_years_experience": 2,
    "required_certs": [],
    "education_evidence_ids": [],
    "min_degree": "bachelor",
    "required_majors": [],
    "major_group": "unknown",
    "accepts_related_field": False,
}


class FakeBackend:
    """Returns queued outputs and records every generate() call."""
    name = "fake"
    model_name = "fake-model"

    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.calls = []

    def generate(self, prompts, schema, max_tokens):
        self.calls.append((prompts, max_tokens))
        if self.rounds:
            return self.rounds.pop(0)
        return [(json.dumps(CV_OK), "stop")] * len(prompts)


def test_schema_requires_every_field_and_forbids_extras():
    for model_cls in (JDEntities, CVEntities):
        schema = json_schema(model_cls)
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["additionalProperties"] is False
    with pytest.raises(ValidationError):
        CVEntities.model_validate({k: v for k, v in CV_OK.items() if k != "major_group"})


def test_validators_reject_wrong_types_and_normalize():
    cv = CVEntities.model_validate(CV_OK)
    assert cv.skills == ["python", "sql"]
    assert JDEntities.model_validate({**JD_OK, "min_years_experience": "3-5 years"}).min_years_experience == 3.0
    for bad in ({"skills": "python"}, {"total_years_experience": "abc"}, {"major_group": "nonsense"}):
        with pytest.raises(ValidationError):
            CVEntities.model_validate({**CV_OK, **bad})


def test_old_cache_entry_is_not_current():
    old_entry = {k: v for k, v in CV_OK.items() if k not in ("majors", "major_group")}
    assert not is_current(CVEntities, old_entry)
    assert not is_current(CVEntities, CV_OK)  # valid fields but no schema version
    assert is_current(CVEntities, {**CV_OK, "_schema_version": SCHEMA_VERSION})


def test_extract_many_isolates_errors_and_retries_once():
    backend = FakeBackend(
        [(json.dumps(CV_OK), "stop"), ("{broken", "stop"), ('{"candidate', "length")],
        [("{still broken", "stop"), (json.dumps(CV_OK), "stop")],
    )
    results = extract_many("cv", ["Education BS Computer Science", "cv two", "cv three"], backend)

    assert [len(p) for p, _ in backend.calls] == [3, 2]
    assert [t for _, t in backend.calls] == [1024, 2048]
    first_prompts, retry_prompts = backend.calls[0][0], backend.calls[1][0]
    assert len(retry_prompts[0]) == len(first_prompts[1]) + 2  # invalid JSON: feedback turns added
    assert retry_prompts[1] == first_prompts[2]  # truncated: same prompt, bigger budget

    assert results[0]["_schema_version"] == SCHEMA_VERSION and results[0]["_backend"] == "fake"
    assert results[0]["major_evidence"] == "Education"  # evidence id [1] -> first segment
    assert isinstance(results[1], str) and results[1].startswith("invalid")
    assert isinstance(results[2], dict)


def test_run_task_sends_one_batched_call_per_chunk():
    script = load_script("02_extract_entities")
    items = {f"k{i}": f"cv {i}" for i in range(10)}
    cache, failed, saves = {}, {"k3": {"error": "old"}}, []
    backend = FakeBackend()

    ok, bad = script.run_task("cv", list(items), items, cache, failed, backend, 4, lambda: saves.append(1))

    assert [len(p) for p, _ in backend.calls] == [4, 4, 2]
    assert (ok, bad, len(cache), len(saves)) == (10, 0, 10, 3)
    assert failed == {}


def test_merge_strict_stops_when_entities_are_missing(tmp_path):
    merge = load_script("03_merge_splits")
    pd.DataFrame([{
        "resume_text": "cv text", "job_description_text": "jd text",
        "label": 1, "job_id": "JD_0000", "label_text": "Potential Fit",
    }]).to_parquet(tmp_path / "val.parquet")
    jd_cache = {"JD_0000": {**JD_OK, "_schema_version": SCHEMA_VERSION}}

    with pytest.raises(SystemExit):
        merge.process_split("val", tmp_path, jd_cache, {})
    assert not (tmp_path / "val.jsonl").exists()

    cv_cache = {compute_text_hash("cv text"): {**CV_OK, "_schema_version": SCHEMA_VERSION}}
    assert merge.process_split("val", tmp_path, jd_cache, cv_cache)["kept"] == 1


def test_evidence_and_segments():
    assert format_evidence_text(["a", "b", "c"], [2, 99, 0, -1]) == "b"
    assert segment_text("Skills: SQL, Excel") == ["Skills", "SQL, Excel"]


def test_atomic_save_and_strict_load(tmp_path):
    path = tmp_path / "cache.json"
    save_json(path, {"a": [1, 2]})
    assert load_json(path) == {"a": [1, 2]}
    assert list(tmp_path.glob("*.tmp*")) == []
    assert load_json(tmp_path / "missing.json", {"d": 1}) == {"d": 1}
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(path)

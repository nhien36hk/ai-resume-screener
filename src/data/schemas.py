import re
from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import DEGREE_LEVELS, MAJOR_GROUPS

SCHEMA_VERSION = 2

DegreeType = Literal["none", "high_school", "associate", "bachelor", "master", "phd"]

MajorGroupType = Literal[
    "computer_science",
    "information_technology",
    "software_engineering",
    "data_science_ai",
    "electrical_electronics",
    "mechanical_engineering",
    "civil_construction",
    "accounting_finance",
    "business_administration",
    "marketing_communications",
    "human_resources",
    "economics",
    "healthcare_medical",
    "biology_biotechnology",
    "chemistry_chemical_engineering",
    "physics_materials_science",
    "mathematics_statistics",
    "law_legal",
    "languages_linguistics",
    "graphic_design_multimedia",
    "supply_chain_logistics",
    "hospitality_tourism",
    "education_pedagogy",
    "environmental_science",
    "general_engineering",
    "other",
    "unknown",
]


def _clean_str_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    seen = set()
    cleaned_items: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.lower().strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            cleaned_items.append(cleaned)
    return cleaned_items


def _clean_cert_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    seen = set()
    cleaned_items: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            cleaned_items.append(cleaned)
    return cleaned_items


def _clean_years(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    if isinstance(val, str):
        match = re.search(r"(\d+(?:\.\d+)?)", val)
        if match:
            return float(match.group(1))
    return 0.0


class JDEntities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_title: str
    must_have_skills: List[str] = Field(default_factory=list, max_length=40)
    nice_to_have_skills: List[str] = Field(default_factory=list, max_length=40)
    min_years_experience: float = Field(default=0.0, ge=0.0)
    min_degree: DegreeType = "none"
    required_certs: List[str] = Field(default_factory=list, max_length=10)
    education_evidence_ids: List[int] = Field(default_factory=list, max_length=3)
    required_majors: List[str] = Field(default_factory=list, max_length=5)
    major_group: MajorGroupType = "unknown"
    accepts_related_field: bool = False

    @field_validator("must_have_skills", "nice_to_have_skills", "required_majors", mode="before")
    @classmethod
    def normalize_str_lists(cls, v: Any) -> List[str]:
        return _clean_str_list(v)

    @field_validator("required_certs", mode="before")
    @classmethod
    def normalize_certs(cls, v: Any) -> List[str]:
        return _clean_cert_list(v)

    @field_validator("min_years_experience", mode="before")
    @classmethod
    def normalize_years(cls, v: Any) -> float:
        return _clean_years(v)

    @field_validator("min_degree", mode="before")
    @classmethod
    def normalize_degree(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower().strip()
        return v

    @field_validator("major_group", mode="before")
    @classmethod
    def normalize_major_group(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower().strip()
        return v


class CVEntities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_title: str
    skills: List[str] = Field(default_factory=list, max_length=40)
    total_years_experience: float = Field(default=0.0, ge=0.0)
    highest_degree: DegreeType = "none"
    certs: List[str] = Field(default_factory=list, max_length=10)
    education_evidence_ids: List[int] = Field(default_factory=list, max_length=3)
    majors: List[str] = Field(default_factory=list, max_length=5)
    major_group: MajorGroupType = "unknown"

    @field_validator("skills", "majors", mode="before")
    @classmethod
    def normalize_str_lists(cls, v: Any) -> List[str]:
        return _clean_str_list(v)

    @field_validator("certs", mode="before")
    @classmethod
    def normalize_certs(cls, v: Any) -> List[str]:
        return _clean_cert_list(v)

    @field_validator("total_years_experience", mode="before")
    @classmethod
    def normalize_years(cls, v: Any) -> float:
        return _clean_years(v)

    @field_validator("highest_degree", mode="before")
    @classmethod
    def normalize_degree(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower().strip()
        return v

    @field_validator("major_group", mode="before")
    @classmethod
    def normalize_major_group(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower().strip()
        return v


def get_jd_json_schema() -> Dict[str, Any]:
    return JDEntities.model_json_schema()


def get_cv_json_schema() -> Dict[str, Any]:
    return CVEntities.model_json_schema()


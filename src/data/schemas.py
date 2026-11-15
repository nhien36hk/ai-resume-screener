"""Pydantic schemas: the single source for the LLM output format and for cache validation."""
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from src.config import DEGREE_LEVELS, MAJOR_GROUPS

SCHEMA_VERSION = 2

Degree = Literal[tuple(DEGREE_LEVELS)]
MajorGroup = Literal[tuple(MAJOR_GROUPS)]


def _str_list(v: Any) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError("expected a list of strings")
    return list(dict.fromkeys(x.strip().lower() for x in v if x.strip()))


def _years(v: Any) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    match = re.match(r"\s*(\d+(?:\.\d+)?)", v) if isinstance(v, str) else None
    if not match:
        raise ValueError("expected a number of years")
    return float(match.group(1))  # "3-5 years" -> 3.0 (lower bound)


StrList = Annotated[list[str], BeforeValidator(_str_list)]
Years = Annotated[float, Field(ge=0), BeforeValidator(_years)]  # Field first, or the schema emits "ge" instead of "minimum"


# Field order matters: the model writes left to right, so evidence comes before the conclusions drawn from it.
class JDEntities(BaseModel):
    job_title: str
    must_have_skills: StrList = Field(max_length=40)
    nice_to_have_skills: StrList = Field(max_length=50)  # 02b dồn must thừa sang nice: pool tối đa 45 trên cache
    min_years_experience: Years
    required_certs: StrList = Field(max_length=10)
    education_evidence_ids: list[int] = Field(max_length=3)
    min_degree: Degree
    required_majors: StrList = Field(max_length=5)
    major_group: MajorGroup
    accepts_related_field: bool


class CVEntities(BaseModel):
    candidate_title: str
    skills: StrList = Field(max_length=128)  # 120B cache: p90=60, max=125 skills per CV
    total_years_experience: Years
    certs: StrList = Field(max_length=20)
    education_evidence_ids: list[int] = Field(max_length=3)
    highest_degree: Degree
    majors: StrList = Field(max_length=10)
    major_group: MajorGroup


def json_schema(model_cls: type[BaseModel]) -> dict:
    schema = model_cls.model_json_schema()
    schema["additionalProperties"] = False  # strict structured output requires it
    return schema


def is_current(model_cls: type[BaseModel], entry: Any) -> bool:
    """True if a cache entry was made with the current schema version and still validates."""
    if not isinstance(entry, dict) or entry.get("_schema_version") != SCHEMA_VERSION:
        return False
    try:
        model_cls.model_validate(entry)
        return True
    except ValidationError:
        return False

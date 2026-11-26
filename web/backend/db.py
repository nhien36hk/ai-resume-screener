"""Quản lý cơ sở dữ liệu SQLite (4 bảng theo Bảng 5.2 Chương 5)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Iterator

from web.backend.config import get_db_path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    required_years REAL NOT NULL DEFAULT 0.0,
    required_degree TEXT NOT NULL DEFAULT 'none',
    major_group TEXT NOT NULL DEFAULT 'unknown',
    extracted_json TEXT NOT NULL,
    jd_cache_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_requirements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    type TEXT NOT NULL,
    is_mandatory INTEGER NOT NULL DEFAULT 1,
    weight REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    masked_contact TEXT NOT NULL,
    sections TEXT NOT NULL DEFAULT '{}',
    extracted TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    error_message TEXT NOT NULL DEFAULT '',
    uploaded_at TEXT NOT NULL,
    UNIQUE(job_id, file_hash),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    raw_score REAL NOT NULL,
    final_score REAL NOT NULL,
    level INTEGER NOT NULL,
    criteria TEXT NOT NULL,
    matched TEXT NOT NULL,
    missing TEXT NOT NULL,
    must_have_failed TEXT NOT NULL,
    best_matches TEXT NOT NULL DEFAULT '[]',
    model_version TEXT NOT NULL,
    elapsed_ms REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, job_id),
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads_json(raw: str, default: Any = None) -> Any:
    if not raw:
        return {} if default is None else default
    return json.loads(raw)

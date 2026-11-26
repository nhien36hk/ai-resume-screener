"""Ứng dụng FastAPI phục vụ đánh giá và xếp hạng mức độ phù hợp CV – JD (Chương 5)."""
from contextlib import asynccontextmanager
import sqlite3
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.backend.config import FRONTEND_DIST_DIR, LEVEL_LABELS, MAX_FILE_SIZE_BYTES
from web.backend.db import dumps_json, get_conn, init_db, loads_json, now_iso
from web.backend.extractor_service import extract_entities, split_sections
from web.backend.parser import FileTooLargeError, UnsupportedFormatError, extract_masked_contact, parse_uploaded_file
from web.backend.scoring_service import ScoringEngine

_ENGINE: ScoringEngine | None = None


def get_engine() -> ScoringEngine:
    if _ENGINE is None:
        raise HTTPException(status_code=503, detail="Mô hình chấm điểm chưa sẵn sàng. Vui lòng thử lại sau giây lát.")
    return _ENGINE


class RequirementUpdateItem(BaseModel):
    text: str = Field(min_length=1)
    type: str = Field(default="skill")
    is_mandatory: bool = Field(default=True)
    weight: float = Field(default=1.0)


class RequirementsPayload(BaseModel):
    requirements: List[RequirementUpdateItem]
    required_years: Optional[float] = None
    required_degree: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ENGINE
    init_db()
    _ENGINE = ScoringEngine()
    yield


app = FastAPI(title="Hệ thống đánh giá mức độ phù hợp CV – JD", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sync_requirements_rows(conn: sqlite3.Connection, job_id: int, extracted: dict) -> List[dict]:
    conn.execute("DELETE FROM job_requirements WHERE job_id = ?", (job_id,))
    rows = []
    for sk in extracted.get("must_have_skills", []):
        rows.append((job_id, sk, "skill", 1, 2.0))
    for sk in extracted.get("nice_to_have_skills", []):
        rows.append((job_id, sk, "skill", 0, 1.0))
    if float(extracted.get("min_years_experience", 0.0)) > 0:
        rows.append((job_id, f"Tối thiểu {extracted['min_years_experience']} năm kinh nghiệm", "experience", 1, 1.0))
    if extracted.get("min_degree", "none") != "none":
        rows.append((job_id, f"Bằng cấp tối thiểu: {extracted['min_degree']}", "education", 0, 1.0))
    for cert in extracted.get("required_certs", []):
        rows.append((job_id, cert, "cert", 0, 1.0))
    conn.executemany(
        "INSERT INTO job_requirements (job_id, text, type, is_mandatory, weight) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return _fetch_requirements(conn, job_id)


def _fetch_requirements(conn: sqlite3.Connection, job_id: int) -> List[dict]:
    cur = conn.execute("SELECT id, job_id, text, type, is_mandatory, weight FROM job_requirements WHERE job_id = ? ORDER BY id", (job_id,))
    return [
        {"id": r["id"], "job_id": r["job_id"], "text": r["text"], "type": r["type"], "is_mandatory": bool(r["is_mandatory"]), "weight": r["weight"]}
        for r in cur.fetchall()
    ]


def _serialize_job(row: sqlite3.Row, reqs: List[dict], cand_count: int = 0) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "description": row["description"],
        "required_years": row["required_years"],
        "required_degree": row["required_degree"],
        "major_group": row["major_group"],
        "extracted": loads_json(row["extracted_json"]),
        "requirements": reqs,
        "candidate_count": cand_count,
        "created_at": row["created_at"],
    }


@app.get("/api/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "model_version": engine.model_version,
        "thresholds": {"t1": engine.cfg["t1"], "t2": engine.cfg["t2"]},
        "weights": engine.cfg["weights"],
        "must_ratio": engine.cfg.get("must_ratio", 1.0),
        "skill_threshold": engine.cfg["skill_threshold"],
    }


@app.get("/api/jobs")
def list_jobs():
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT j.*, COUNT(c.id) AS cand_count FROM jobs j LEFT JOIN candidates c ON c.job_id = j.id "
            "GROUP BY j.id ORDER BY j.id DESC"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "required_years": r["required_years"],
                "required_degree": r["required_degree"],
                "candidate_count": r["cand_count"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


@app.post("/api/jobs")
async def create_job(
    request: Request,
    title: Optional[str] = Form(default=None),
    description: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    engine = get_engine()
    jd_title, jd_desc = title or "", description or ""
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        body = await request.json()
        jd_title = str(body.get("title", "")).strip()
        jd_desc = str(body.get("description", "")).strip()
    elif file is not None:
        raw_bytes = await file.read()
        try:
            parsed = parse_uploaded_file(file.filename or "jd.pdf", raw_bytes)
        except UnsupportedFormatError as exc:
            raise HTTPException(status_code=415, detail=str(exc))
        except FileTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        if parsed["status"] != "ok":
            raise HTTPException(status_code=422, detail=parsed["error_message"])
        jd_desc = parsed["raw_text"]

    if len(jd_desc.strip()) < 20:
        raise HTTPException(status_code=400, detail="Mô tả công việc (JD) quá ngắn hoặc bị trống. Vui lòng nhập ít nhất 20 ký tự.")

    extracted, source = extract_entities("jd", jd_desc, title=jd_title)
    final_title = jd_title.strip() or extracted.get("job_title") or "Vị trí tuyển dụng"
    extracted["job_title"] = final_title
    extracted["_source"] = source
    jd_cache = engine.build_jd_cache(jd_desc, extracted.get("must_have_skills", []), extracted.get("nice_to_have_skills", []))

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (title, description, required_years, required_degree, major_group, extracted_json, jd_cache_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                final_title,
                jd_desc,
                float(extracted.get("min_years_experience", 0.0)),
                extracted.get("min_degree", "none"),
                extracted.get("major_group", "unknown"),
                dumps_json(extracted),
                dumps_json(jd_cache),
                now_iso(),
            ),
        )
        job_id = int(cur.lastrowid)
        reqs = _sync_requirements_rows(conn, job_id, extracted)
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return JSONResponse(status_code=201, content=_serialize_job(row, reqs, 0))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí tuyển dụng #{job_id}.")
        reqs = _fetch_requirements(conn, job_id)
        cnt = conn.execute("SELECT COUNT(*) FROM candidates WHERE job_id = ?", (job_id,)).fetchone()[0]
        return _serialize_job(row, reqs, int(cnt))


@app.put("/api/jobs/{job_id}/requirements")
def update_requirements(job_id: int, payload: RequirementsPayload):
    engine = get_engine()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí tuyển dụng #{job_id}.")

        extracted = loads_json(row["extracted_json"])
        must_skills, nice_skills, certs = [], [], []
        conn.execute("DELETE FROM job_requirements WHERE job_id = ?", (job_id,))
        for item in payload.requirements:
            w = 2.0 if (item.type == "skill" and item.is_mandatory) else item.weight
            conn.execute(
                "INSERT INTO job_requirements (job_id, text, type, is_mandatory, weight) VALUES (?, ?, ?, ?, ?)",
                (job_id, item.text.strip(), item.type, 1 if item.is_mandatory else 0, w),
            )
            if item.type == "skill":
                (must_skills if item.is_mandatory else nice_skills).append(item.text.strip().lower())
            elif item.type == "cert":
                certs.append(item.text.strip().lower())

        extracted["must_have_skills"] = list(dict.fromkeys(must_skills))
        extracted["nice_to_have_skills"] = [s for s in dict.fromkeys(nice_skills) if s not in set(extracted["must_have_skills"])]
        if certs:
            extracted["required_certs"] = list(dict.fromkeys(certs))
        req_years = float(payload.required_years) if payload.required_years is not None else float(row["required_years"])
        req_degree = payload.required_degree if payload.required_degree is not None else row["required_degree"]
        extracted["min_years_experience"] = req_years
        extracted["min_degree"] = req_degree

        jd_cache = engine.build_jd_cache(row["description"], extracted["must_have_skills"], extracted["nice_to_have_skills"])
        conn.execute(
            "UPDATE jobs SET required_years = ?, required_degree = ?, extracted_json = ?, jd_cache_json = ? WHERE id = ?",
            (req_years, req_degree, dumps_json(extracted), dumps_json(jd_cache), job_id),
        )
        updated = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        reqs = _fetch_requirements(conn, job_id)
        cnt = conn.execute("SELECT COUNT(*) FROM candidates WHERE job_id = ?", (job_id,)).fetchone()[0]
        return _serialize_job(updated, reqs, int(cnt))


@app.post("/api/jobs/{job_id}/cvs")
async def upload_cvs(job_id: int, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Chưa chọn tệp CV nào để tải lên.")
    with get_conn() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí tuyển dụng #{job_id}.")

    results: List[dict] = []
    single_file = len(files) == 1
    for upload in files:
        fname = upload.filename or "cv.pdf"
        content = await upload.read()
        try:
            parsed = parse_uploaded_file(fname, content)
        except UnsupportedFormatError as exc:
            if single_file:
                raise HTTPException(status_code=415, detail=str(exc))
            results.append({"file_name": fname, "status": "error", "error_message": str(exc)})
            continue
        except FileTooLargeError as exc:
            if single_file:
                raise HTTPException(status_code=413, detail=str(exc))
            results.append({"file_name": fname, "status": "error", "error_message": str(exc)})
            continue

        with get_conn() as conn:
            existing = conn.execute(
                "SELECT id, file_name, status FROM candidates WHERE job_id = ? AND file_hash = ?",
                (job_id, parsed["file_hash"]),
            ).fetchone()
            if existing:
                results.append({
                    "candidate_id": existing["id"],
                    "file_name": fname,
                    "file_hash": parsed["file_hash"],
                    "status": "duplicate",
                    "error_message": f"Trùng nội dung với hồ sơ đã tải ({existing['file_name']}).",
                })
                continue

            if parsed["status"] == "ok":
                extracted, source = extract_entities("cv", parsed["raw_text"])
                extracted["_source"] = source
                sections = split_sections(parsed["raw_text"])
                masked_contact = extract_masked_contact(parsed["raw_text"], fname)
            else:
                extracted, sections, masked_contact = {}, {}, extract_masked_contact("", fname)

            cur = conn.execute(
                "INSERT INTO candidates (job_id, file_name, file_hash, raw_text, masked_contact, sections, extracted, status, error_message, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    fname,
                    parsed["file_hash"],
                    parsed["raw_text"],
                    masked_contact,
                    dumps_json(sections),
                    dumps_json(extracted),
                    parsed["status"],
                    parsed["error_message"],
                    now_iso(),
                ),
            )
            results.append({
                "candidate_id": int(cur.lastrowid),
                "file_name": fname,
                "file_hash": parsed["file_hash"],
                "status": parsed["status"],
                "masked_contact": masked_contact,
                "error_message": parsed["error_message"],
            })
    return {"job_id": job_id, "uploaded": results}


@app.get("/api/jobs/{job_id}/cvs")
def list_job_cvs(job_id: int):
    with get_conn() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí tuyển dụng #{job_id}.")
        rows = conn.execute(
            "SELECT id, file_name, file_hash, masked_contact, status, error_message, uploaded_at "
            "FROM candidates WHERE job_id = ? ORDER BY id DESC",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/jobs/{job_id}/evaluate")
def evaluate_job(job_id: int):
    engine = get_engine()
    with get_conn() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí tuyển dụng #{job_id}.")
        cand_rows = conn.execute("SELECT * FROM candidates WHERE job_id = ? AND status = 'ok' ORDER BY id", (job_id,)).fetchall()
        if not cand_rows:
            return {"job_id": job_id, "evaluated_count": 0, "timings": {}, "results": []}

        jd_extracted = loads_json(job["extracted_json"])
        jd_cache = loads_json(job["jd_cache_json"])
        candidates = [{"id": r["id"], "raw_text": r["raw_text"], "extracted": loads_json(r["extracted"])} for r in cand_rows]

    eval_outputs, timings = engine.evaluate_candidates_batch(jd_extracted, jd_cache, candidates)

    with get_conn() as conn:
        for res in eval_outputs:
            conn.execute(
                """
                INSERT INTO evaluations (
                    candidate_id, job_id, raw_score, final_score, level, criteria, matched, missing,
                    must_have_failed, best_matches, model_version, elapsed_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, job_id) DO UPDATE SET
                    raw_score=excluded.raw_score,
                    final_score=excluded.final_score,
                    level=excluded.level,
                    criteria=excluded.criteria,
                    matched=excluded.matched,
                    missing=excluded.missing,
                    must_have_failed=excluded.must_have_failed,
                    best_matches=excluded.best_matches,
                    model_version=excluded.model_version,
                    elapsed_ms=excluded.elapsed_ms,
                    created_at=excluded.created_at
                """,
                (
                    res["candidate_id"],
                    job_id,
                    res["raw_score"],
                    res["final_score"],
                    res["level"],
                    dumps_json(res["criteria"]),
                    dumps_json(res["matched"]),
                    dumps_json(res["missing"]),
                    dumps_json(res["must_have_failed"]),
                    dumps_json(res["best_matches"]),
                    res["model_version"],
                    res["elapsed_ms"],
                    now_iso(),
                ),
            )
    return {"job_id": job_id, "evaluated_count": len(eval_outputs), "timings": timings, "results": eval_outputs}


@app.get("/api/jobs/{job_id}/ranking")
def get_ranking(job_id: int, top_k: int = 20, level: Optional[int] = None):
    with get_conn() as conn:
        job = conn.execute("SELECT id, title FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy vị trí tuyển dụng #{job_id}.")
        sql = """
            SELECT e.*, c.file_name, c.masked_contact, c.status AS candidate_status
            FROM evaluations e
            JOIN candidates c ON c.id = e.candidate_id
            WHERE e.job_id = ?
        """
        params: List[Any] = [job_id]
        if level is not None:
            sql += " AND e.level = ?"
            params.append(level)
        sql += " ORDER BY e.final_score DESC, e.raw_score DESC, e.id ASC LIMIT ?"
        params.append(max(1, top_k))
        rows = conn.execute(sql, params).fetchall()
        ranking = []
        for idx, r in enumerate(rows, 1):
            ranking.append({
                "rank": idx,
                "eval_id": r["id"],
                "candidate_id": r["candidate_id"],
                "file_name": r["file_name"],
                "masked_contact": r["masked_contact"],
                "raw_score": r["raw_score"],
                "final_score": r["final_score"],
                "level": r["level"],
                "level_label": LEVEL_LABELS.get(r["level"], "Không rõ"),
                "criteria": loads_json(r["criteria"]),
                "must_have_failed": loads_json(r["must_have_failed"], []),
                "model_version": r["model_version"],
                "elapsed_ms": r["elapsed_ms"],
            })
        return {"job_id": job_id, "job_title": job["title"], "total": len(ranking), "items": ranking}


@app.get("/api/evaluations/{eval_id}")
def get_evaluation_detail(eval_id: int):
    engine = get_engine()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT e.*, c.file_name, c.masked_contact, c.extracted, c.sections, j.title AS job_title, j.extracted_json
            FROM evaluations e
            JOIN candidates c ON c.id = e.candidate_id
            JOIN jobs j ON j.id = e.job_id
            WHERE e.id = ?
            """,
            (eval_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy kết quả đánh giá #{eval_id}.")
        return {
            "eval_id": row["id"],
            "candidate_id": row["candidate_id"],
            "job_id": row["job_id"],
            "job_title": row["job_title"],
            "file_name": row["file_name"],
            "masked_contact": row["masked_contact"],
            "raw_score": row["raw_score"],
            "final_score": row["final_score"],
            "level": row["level"],
            "level_label": LEVEL_LABELS.get(row["level"], "Không rõ"),
            "thresholds": {"t1": engine.cfg["t1"], "t2": engine.cfg["t2"]},
            "weights": engine.cfg["weights"],
            "criteria": loads_json(row["criteria"]),
            "matched": loads_json(row["matched"], []),
            "missing": loads_json(row["missing"], []),
            "must_have_failed": loads_json(row["must_have_failed"], []),
            "best_matches": loads_json(row["best_matches"], []),
            "candidate_extracted": loads_json(row["extracted"]),
            "jd_extracted": loads_json(row["extracted_json"]),
            "model_version": row["model_version"],
            "elapsed_ms": row["elapsed_ms"],
            "created_at": row["created_at"],
        }


@app.delete("/api/candidates/{candidate_id}")
def delete_candidate(candidate_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ứng viên #{candidate_id}.")
        conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        return {"deleted": True, "candidate_id": candidate_id}


if FRONTEND_DIST_DIR.exists():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Không tìm thấy đường dẫn API.")
        target = FRONTEND_DIST_DIR / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")

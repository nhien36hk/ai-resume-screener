"""Kiểm thử toàn diện API hệ thống web Chương 5 (TC01–TC10, xóa hồ sơ, kiểm tra công bằng §5.8)."""
import io
from pathlib import Path

import docx
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from web.backend.config import MAX_FILE_SIZE_BYTES
from web.backend.main import app

SAMPLE_JD = (
    "Position: Senior Data Analyst\n"
    "Required: 3+ years of experience in data analysis.\n"
    "Must have skills: Python, SQL, Tableau, Excel, data modeling.\n"
    "Preferred: Power BI, ERP.\n"
    "Bachelor degree in Computer Science or Data Science."
)


def make_text_pdf(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for ln in lines:
        c.drawString(50, y, ln)
        y -= 18
    c.save()
    return buf.getvalue()


def make_scanned_pdf() -> bytes:
    img = Image.new("RGB", (200, 100), color="white")
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=letter)
    c.drawImage(ImageReader(img_buf), 50, 600, width=200, height=100)
    c.save()
    return pdf_buf.getvalue()


def make_docx_bytes(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table_rows:
        tbl = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for i, row in enumerate(table_rows):
            for j, val in enumerate(row):
                tbl.cell(i, j).text = val
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def web_client(tmp_path_factory, monkeypatch_module=None):
    db_file = tmp_path_factory.mktemp("webdb") / "test_screening.db"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("WEB_DB_PATH", str(db_file))
        mp.setenv("WEB_DISABLE_API", "1")
        with TestClient(app) as client:
            yield client


def test_health_and_tc01_create_job_and_edit_reqs(web_client: TestClient):
    h = web_client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    # TC01: Tạo JD hợp lệ
    res = web_client.post("/api/jobs", json={"title": "Data Analyst", "description": SAMPLE_JD})
    assert res.status_code == 201
    data = res.json()
    assert data["id"] > 0
    assert len(data["requirements"]) >= 4

    # Chỉnh sửa yêu cầu bắt buộc (PUT /api/jobs/{job_id}/requirements)
    reqs = data["requirements"]
    upd = web_client.put(
        f"/api/jobs/{data['id']}/requirements",
        json={"requirements": reqs, "required_years": 3.0, "required_degree": "bachelor"},
    )
    assert upd.status_code == 200
    assert upd.json()["required_years"] == 3.0


def test_tc02_to_tc05_file_parsing_and_validation(web_client: TestClient):
    job_id = web_client.post("/api/jobs", json={"title": "Data Analyst", "description": SAMPLE_JD}).json()["id"]

    # TC02: Tải 1 PDF có lớp chữ
    pdf_ok = make_text_pdf([
        "Summary",
        "Data Analyst with 4 years of experience in business intelligence.",
        "Skills: Python, SQL, Tableau, Excel, data modeling, Power BI.",
        "Education: Bachelor of Computer Science.",
    ])
    r2 = web_client.post(f"/api/jobs/{job_id}/cvs", files=[("files", ("good_cv.pdf", pdf_ok, "application/pdf"))])
    assert r2.status_code == 200
    assert r2.json()["uploaded"][0]["status"] == "ok"

    # TC03: Tải tệp sai định dạng (.png) -> HTTP 415
    r3 = web_client.post(f"/api/jobs/{job_id}/cvs", files=[("files", ("bad.png", b"fake-png", "image/png"))])
    assert r3.status_code == 415

    # TC04: Tải tệp vượt kích thước 5 MB -> HTTP 413
    oversized = b"%PDF-1.4\n" + b"0" * (MAX_FILE_SIZE_BYTES + 128)
    r4 = web_client.post(f"/api/jobs/{job_id}/cvs", files=[("files", ("huge.pdf", oversized, "application/pdf"))])
    assert r4.status_code == 413

    # TC05: PDF ảnh quét không có lớp chữ -> status unreadable, không chấm điểm
    r5 = web_client.post(f"/api/jobs/{job_id}/cvs", files=[("files", ("scan.pdf", make_scanned_pdf(), "application/pdf"))])
    assert r5.status_code == 200
    assert r5.json()["uploaded"][0]["status"] == "unreadable"

    # TC09: Tải lại đúng PDF đã có -> phát hiện trùng theo file_hash
    r9 = web_client.post(f"/api/jobs/{job_id}/cvs", files=[("files", ("good_cv_copy.pdf", pdf_ok, "application/pdf"))])
    assert r9.status_code == 200
    assert r9.json()["uploaded"][0]["status"] == "duplicate"
    cvs = web_client.get(f"/api/jobs/{job_id}/cvs").json()
    assert len([c for c in cvs if c["status"] == "ok"]) == 1


def test_tc06_batch_50_tc07_detail_tc08_must_have_tc10_deterministic(web_client: TestClient):
    job_id = web_client.post("/api/jobs", json={"title": "Data Analyst", "description": SAMPLE_JD}).json()["id"]

    # Tạo 49 CV DOCX + 1 CV thiếu điều kiện bắt buộc (TC08) = 50 CV (TC06)
    batch_files = []
    for i in range(49):
        doc_bytes = make_docx_bytes(
            [
                f"Candidate {i} | email{i}@example.com",
                "Summary",
                f"Experienced Data Analyst with {2 + (i % 5)} years working with Python, SQL, Tableau, Excel and data modeling.",
                "Education: Bachelor in Computer Science.",
            ],
            table_rows=[["Skill", "Level"], ["Python, SQL, Tableau", "Advanced"]],
        )
        batch_files.append(("files", (f"cv_{i:02d}.docx", doc_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")))

    # CV thiếu kỹ năng bắt buộc và kinh nghiệm thấp (TC08)
    weak_pdf = make_text_pdf([
        "Summary",
        "Office assistant with 1 year experience in writing reports and organizing meetings.",
        "Skills: Microsoft Word, communication, scheduling.",
        "Education: Bachelor of Arts.",
    ])
    batch_files.append(("files", ("missing_must.pdf", weak_pdf, "application/pdf")))

    up = web_client.post(f"/api/jobs/{job_id}/cvs", files=batch_files)
    assert up.status_code == 200
    assert len(up.json()["uploaded"]) == 50

    ev1 = web_client.post(f"/api/jobs/{job_id}/evaluate")
    assert ev1.status_code == 200
    assert ev1.json()["evaluated_count"] == 50

    # TC06: Đủ 50 CV và sắp xếp đúng theo điểm giảm dần
    rk = web_client.get(f"/api/jobs/{job_id}/ranking?top_k=60").json()
    assert rk["total"] == 50
    scores = [(r["final_score"], r["raw_score"]) for r in rk["items"]]
    assert scores == sorted(scores, reverse=True)

    # TC07: Chi tiết kết quả đầu bảng
    top_item = rk["items"][0]
    detail = web_client.get(f"/api/evaluations/{top_item['eval_id']}").json()
    assert set(detail["criteria"].keys()) == {"sem", "skill", "exp", "edu", "cert"}
    assert len(detail["best_matches"]) >= 1
    assert isinstance(detail["matched"], list) and isinstance(detail["missing"], list)

    # TC08: CV thiếu kỹ năng bắt buộc bị giới hạn dưới ngưỡng t2 và ghi lý do
    weak_item = next(r for r in rk["items"] if r["file_name"] == "missing_must.pdf")
    assert len(weak_item["must_have_failed"]) >= 1
    assert weak_item["final_score"] < detail["thresholds"]["t2"]
    assert weak_item["level"] < 2

    # TC10: Chấm lại cùng tập CV & JD cho kết quả giống hệt nhau
    ev2 = web_client.post(f"/api/jobs/{job_id}/evaluate").json()
    rk2 = web_client.get(f"/api/jobs/{job_id}/ranking?top_k=60").json()
    assert [(r["final_score"], r["level"]) for r in rk2["items"]] == [(r["final_score"], r["level"]) for r in rk["items"]]
    assert ev2["evaluated_count"] == 50

    # Xóa 1 ứng viên (§5.8)
    del_res = web_client.delete(f"/api/candidates/{weak_item['candidate_id']}")
    assert del_res.status_code == 200
    assert web_client.get(f"/api/jobs/{job_id}/ranking?top_k=60").json()["total"] == 49


def test_fairness_name_and_dob_invariance(web_client: TestClient):
    """Mục 5.8: Đổi tên nam/nữ và thêm/bớt ngày sinh ở đầu CV không làm thay đổi điểm."""
    job_id = web_client.post("/api/jobs", json={"title": "Data Analyst", "description": SAMPLE_JD}).json()["id"]
    body = [
        "Summary",
        "Experienced Data Analyst with 4 years working on Python, SQL, Tableau, Excel and data modeling.",
        "Education: Bachelor of Data Science.",
    ]
    cv_male_en = make_text_pdf(["John Smith", "Email: john@example.com", *body])
    cv_female_dob_en = make_text_pdf(["Emily Smith", "Date of Birth: 15/04/1998", "Phone: +84 912345678", *body])
    cv_male_vi = make_text_pdf(["Họ tên: Nguyễn Văn Nam", "Ngày sinh: 15/04/1998", *body])
    cv_female_vi = make_text_pdf(["Họ tên: Nguyễn Thị Lan", *body])

    files = [
        ("files", ("male_en.pdf", cv_male_en, "application/pdf")),
        ("files", ("female_dob_en.pdf", cv_female_dob_en, "application/pdf")),
        ("files", ("male_vi.pdf", cv_male_vi, "application/pdf")),
        ("files", ("female_vi.pdf", cv_female_vi, "application/pdf")),
    ]
    up = web_client.post(f"/api/jobs/{job_id}/cvs", files=files)
    assert up.status_code == 200
    ev = web_client.post(f"/api/jobs/{job_id}/evaluate").json()
    results = ev["results"]
    assert len(results) == 4
    base = results[0]
    for other in results[1:]:
        assert abs(other["final_score"] - base["final_score"]) < 1e-9
        assert abs(other["raw_score"] - base["raw_score"]) < 1e-9
        assert other["criteria"] == base["criteria"]
        assert other["level"] == base["level"]

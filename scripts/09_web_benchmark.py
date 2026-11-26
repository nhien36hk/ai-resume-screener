"""Đo hiệu năng hệ thống web qua API (§5.6) và kiểm thử bộ 20 tệp PDF/DOCX đa dạng (§5.7 V6-b)."""
import argparse
import io
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import docx
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.config import RESULTS_DIR
from src.data.loader import load_all
from src.utils import save_json, to_markdown_table


def _pdf_from_text(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 760
    for line in text.replace("\r", "").split("\n")[:40]:
        safe = line[:95].encode("latin-1", "replace").decode("latin-1")
        c.drawString(40, y, safe)
        y -= 16
        if y < 50:
            c.showPage()
            y = 760
    c.save()
    return buf.getvalue()


def _docx_from_text(text: str, with_table: bool = False) -> bytes:
    d = docx.Document()
    for line in text.replace("\r", "").split("\n")[:40]:
        if line.strip():
            d.add_paragraph(line.strip())
    if with_table:
        tbl = d.add_table(rows=2, cols=2)
        tbl.cell(0, 0).text = "Kỹ năng chính"
        tbl.cell(0, 1).text = "Python, SQL, Excel"
        tbl.cell(1, 0).text = "Kinh nghiệm"
        tbl.cell(1, 1).text = "3 năm"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _scanned_pdf() -> bytes:
    img = Image.new("RGB", (240, 120), color="white")
    ib = io.BytesIO()
    img.save(ib, format="PNG")
    ib.seek(0)
    pb = io.BytesIO()
    c = canvas.Canvas(pb, pagesize=letter)
    c.drawImage(ImageReader(ib), 50, 600, width=240, height=120)
    c.save()
    return pb.getvalue()


def run_benchmark_and_file_tests() -> None:
    df = load_all()
    jd_row = df.iloc[0]
    unique_cvs = df.drop_duplicates("resume_text").head(100)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["WEB_DB_PATH"] = str(Path(tmpdir) / "bench.db")
        os.environ["WEB_DISABLE_API"] = "1"
        from web.backend.main import app

        with TestClient(app) as client:
            # 1. Warmup
            j_warm = client.post("/api/jobs", json={"title": "Warmup", "description": jd_row["job_description_text"]}).json()
            warm_pdf = _pdf_from_text(unique_cvs.iloc[0]["resume_text"])
            client.post(f"/api/jobs/{j_warm['id']}/cvs", files=[("files", ("warm.pdf", warm_pdf, "application/pdf"))])
            client.post(f"/api/jobs/{j_warm['id']}/evaluate")

            # 2. Benchmark 1, 10, 50, 100 CV qua API (PDF thật)
            bench_records = []
            pdf_cache = [
                (f"cv_{i:03d}.pdf", _pdf_from_text(txt))
                for i, txt in enumerate(unique_cvs["resume_text"])
            ]
            for n_cv in (1, 10, 50, 100):
                job = client.post("/api/jobs", json={"title": f"Bench {n_cv}", "description": jd_row["job_description_text"]}).json()
                jid = job["id"]
                t_up0 = time.perf_counter()
                files = [("files", (fn, b, "application/pdf")) for fn, b in pdf_cache[:n_cv]]
                up_res = client.post(f"/api/jobs/{jid}/cvs", files=files)
                assert up_res.status_code == 200
                upload_parse_ms = (time.perf_counter() - t_up0) * 1000.0

                t_ev0 = time.perf_counter()
                ev_res = client.post(f"/api/jobs/{jid}/evaluate")
                assert ev_res.status_code == 200
                ev_data = ev_res.json()
                eval_api_ms = (time.perf_counter() - t_ev0) * 1000.0
                total_ms = upload_parse_ms + eval_api_ms
                bench_records.append({
                    "n_cv": n_cv,
                    "upload_parse_ms": round(upload_parse_ms, 1),
                    "eval_api_ms": round(eval_api_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "per_cv_ms": round(total_ms / n_cv, 1),
                    "timings": ev_data["timings"],
                })

            headers = ["Số CV", "Tải + đọc PDF + trích xuất (ms)", "Chấm điểm + lưu CSDL (ms)", "Tổng qua API (ms)", "TB mỗi CV (ms)"]
            rows = [[str(r["n_cv"]), f"{r['upload_parse_ms']:.1f}", f"{r['eval_api_ms']:.1f}", f"{r['total_ms']:.1f}", f"{r['per_cv_ms']:.1f}"] for r in bench_records]
            md = "# Hiệu năng hệ thống web qua API (E6 toàn trình, §5.6)\n\n" + to_markdown_table(headers, rows) + "\n"
            save_json(RESULTS_DIR / "web_e6.json", {"benchmarks": bench_records})
            (RESULTS_DIR / "web_e6.md").write_text(md, encoding="utf-8")
            print(md)

            # 3. Kiểm thử đọc 20 tệp thật đa dạng (§5.7 phương án V6-b)
            j_file = client.post("/api/jobs", json={"title": "Data Analyst", "description": jd_row["job_description_text"]}).json()["id"]
            test_cases = []
            for i in range(8):
                test_cases.append(("pdf_en", f"en_{i}.pdf", _pdf_from_text(unique_cvs.iloc[i]["resume_text"]), "application/pdf"))
            for i in range(4):
                vi_txt = f"Tóm tắt\nChuyên viên phân tích dữ liệu {i+2} năm kinh nghiệm Python, SQL, Tableau, Excel.\nHọc vấn: Cử nhân Khoa học dữ liệu."
                test_cases.append(("docx_vi_table", f"vi_{i}.docx", _docx_from_text(vi_txt, with_table=True), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            for i in range(4):
                test_cases.append(("docx_en", f"en_{i}.docx", _docx_from_text(unique_cvs.iloc[i + 10]["resume_text"]), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
            for i in range(2):
                test_cases.append(("pdf_scanned", f"scan_{i}.pdf", _scanned_pdf(), "application/pdf"))
            for i in range(2):
                test_cases.append(("invalid_png", f"img_{i}.png", b"PNGDATA", "image/png"))

            file_reports = []
            for kind, fname, raw_b, mime in test_cases:
                resp = client.post(f"/api/jobs/{j_file}/cvs", files=[("files", (fname, raw_b, mime))])
                if resp.status_code == 200:
                    st = resp.json()["uploaded"][0]["status"]
                    msg = resp.json()["uploaded"][0]["error_message"]
                else:
                    st = f"http_{resp.status_code}"
                    msg = resp.json().get("detail", "")
                file_reports.append({"kind": kind, "file_name": fname, "status": st, "detail": msg})

            save_json(RESULTS_DIR / "web_file_test.json", {"total": len(file_reports), "results": file_reports})
            print(f"Đã kiểm thử {len(file_reports)} tệp PDF/DOCX/ảnh/sai định dạng -> {RESULTS_DIR / 'web_file_test.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_benchmark_and_file_tests()

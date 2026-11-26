"""Đọc văn bản từ PDF/DOCX và che thông tin cá nhân khi hiển thị."""
import hashlib
import io
from pathlib import Path
import re

import docx
import pdfplumber

from web.backend.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MIN_TEXT_CHARS

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")


class UnsupportedFormatError(Exception):
    """Tệp không thuộc định dạng PDF hoặc DOCX."""


class FileTooLargeError(Exception):
    """Tệp vượt quá dung lượng cho phép."""


def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def mask_pii_for_display(text: str) -> str:
    masked = EMAIL_RE.sub("[EMAIL ĐÃ ẨN]", text)
    masked = URL_RE.sub("[LIÊN KẾT ĐÃ ẨN]", masked)
    masked = PHONE_RE.sub("[SĐT ĐÃ ẨN]", masked)
    return masked


def extract_masked_contact(text: str, file_name: str) -> str:
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    header_hint = mask_pii_for_display(" | ".join(lines[:2])) if lines else ""
    stem = Path(file_name).stem
    return f"{stem} ({header_hint[:80]})" if header_hint else stem


def _read_pdf_bytes(content: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt.strip())
    return "\n".join(pages)


def _read_docx_bytes(content: bytes) -> str:
    document = docx.Document(io.BytesIO(content))
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            row_cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if row_cells:
                parts.append(" | ".join(dict.fromkeys(row_cells)))
    return "\n".join(parts)


def parse_uploaded_file(file_name: str, content: bytes) -> dict:
    """Đọc PDF/DOCX và trả về dict gồm text, file_hash, status, error_message."""
    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Định dạng '{ext or 'không rõ'}' chưa được hỗ trợ. Vui lòng tải tệp .pdf hoặc .docx."
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise FileTooLargeError(f"Tệp '{file_name}' vượt quá giới hạn tối đa {mb} MB.")

    file_hash = compute_file_hash(content)
    try:
        raw_text = _read_pdf_bytes(content) if ext == ".pdf" else _read_docx_bytes(content)
    except Exception as exc:
        return {
            "file_hash": file_hash,
            "raw_text": "",
            "status": "unreadable",
            "error_message": f"Không thể đọc tệp ({exc})",
        }

    cleaned = re.sub(r"[ \t]+", " ", raw_text.replace("\r", "")).strip()
    if len(cleaned) < MIN_TEXT_CHARS:
        return {
            "file_hash": file_hash,
            "raw_text": cleaned,
            "status": "unreadable",
            "error_message": "Không trích được văn bản (tệp rỗng hoặc PDF dạng ảnh quét).",
        }
    return {
        "file_hash": file_hash,
        "raw_text": cleaned,
        "status": "ok",
        "error_message": "",
    }

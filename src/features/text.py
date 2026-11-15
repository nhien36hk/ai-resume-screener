import re
from typing import Any, List, Tuple

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CV_HEADERS

H_PATTERN = "|".join(sorted(map(re.escape, CV_HEADERS), key=len, reverse=True))
HEADER_RE = re.compile(rf"(?<![A-Za-z])({H_PATTERN})(?=[A-Z0-9:\s-]):?")
SENT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z])|\s+[-•●▪]\s*(?=[A-Z])")
MAX_SEGMENT_WORDS = 30


def segment_text(text: str) -> List[str]:
    text = HEADER_RE.sub(r"\n\1\n", text)
    segs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in SENT_RE.split(line):
            s = s.strip()
            if not s:
                continue
            words = s.split()
            for i in range(0, len(words), MAX_SEGMENT_WORDS):
                chunk = " ".join(words[i:i + MAX_SEGMENT_WORDS])
                if chunk:
                    segs.append(chunk)
    return segs


def number_segments(text: str) -> Tuple[str, List[str]]:
    segs = segment_text(text)
    numbered_lines = [f"[{i}] {s}" for i, s in enumerate(segs, 1)]
    return "\n".join(numbered_lines), segs


def chunk_text(text: str, tokenizer: Any, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chia nhỏ văn bản (CV hoặc JD) thành các đoạn token với độ chồng lấn overlap."""
    if not text or not text.strip():
        return [""]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= chunk_size: 
        return [text.strip()]

    stride = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(tokens), stride):
        sub_tokens = tokens[i:i + chunk_size]
        chunk_str = tokenizer.decode(sub_tokens, skip_special_tokens=True).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if i + chunk_size >= len(tokens):
            break
    return chunks or [text.strip()]



def prepare_finetune_text(text: str) -> str:
    """Loại bỏ thông tin cá nhân ở đầu CV, giữ lại từ phần Skills/Experience/Summary."""
    if not text:
        return ""
    match = re.search(rf"\b({H_PATTERN})\b", text, re.IGNORECASE)
    if match and match.start() > 0:
        return text[match.start():].strip()
    return text.strip()


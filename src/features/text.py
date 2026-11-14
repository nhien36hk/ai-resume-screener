import re
from typing import List, Tuple

from src.config import CV_HEADERS

H_PATTERN = "|".join(sorted(map(re.escape, CV_HEADERS), key=len, reverse=True))
HEADER_RE = re.compile(rf"(?<![A-Za-z])({H_PATTERN})(?=[A-Z0-9:\s-]):?")  # drop the colon after a header so the next segment does not start with ":"
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


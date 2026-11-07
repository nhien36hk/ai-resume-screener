# Level 1: Thư viện chuẩn Python (Standard Library)
import re
from typing import List, Optional

# Level 2: Thư viện bên thứ ba (Third-party Packages)
from transformers import AutoTokenizer

# Level 3: Module nội bộ trong dự án (Project-Local Imports)


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 50


class TextChunker:
    """Xử lý chia nhỏ văn bản CV thành chunks và tách câu yêu cầu trong JD."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stride = chunk_size - chunk_overlap

    def chunk_cv(self, text: str) -> List[str]:
        """Chia văn bản CV thành các chunk tối đa chunk_size tokens, overlap chunk_overlap tokens."""
        if not text or not text.strip():
            return [""]

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(tokens), self.stride):
            window = tokens[i : i + self.chunk_size]
            chunk_text = self.tokenizer.decode(window, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if i + self.chunk_size >= len(tokens):
                break

        return chunks if chunks else [text]

    @staticmethod
    def split_jd_requirements(text: str) -> List[str]:
        """Tách văn bản JD thành danh sách các câu / ý yêu cầu độc lập."""
        if not text or not text.strip():
            return [""]

        # Tách theo dòng mới hoặc các dấu chấm kết thúc câu
        raw_parts = re.split(r"[\n\r]+|[•\-\*]\s+|(?<=[.!?])\s+", text)
        sentences = [p.strip() for p in raw_parts if len(p.strip()) >= 5]
        return sentences if sentences else [text.strip()]


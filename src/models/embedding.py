# Level 1: Thư viện chuẩn Python (Standard Library)
import os
from typing import List, Optional, Tuple, Union

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Level 3: Module nội bộ trong dự án (Project-Local Imports)
from src.data.text_chunker import TextChunker

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SemanticEmbeddingPipeline:
    """Mô hình nhúng và tính toán điểm tương đồng ngữ nghĩa (S_sem) theo chuẩn Mục 2 plan.md."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = DEFAULT_DEVICE,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.model.max_seq_length = max_seq_length
        self.chunker = TextChunker(model_name=model_name_or_path, chunk_size=200, chunk_overlap=50)

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Trích xuất vector nhúng chuẩn hóa L2 (normalize_embeddings=True) để dùng dot-product."""
        if not sentences:
            return np.zeros((1, 384), dtype=np.float32)

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            device=self.device,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def compute_pair_similarity_chunked(self, jd_text: str, cv_text: str) -> float:
        """
        Tính điểm tương đồng giữa 1 JD và 1 CV dùng kỹ thuật Chunking (E3b):
        sim(c, j) = (1 / |R_j|) * ∑_{r ∈ R_j} max_{p ∈ P_c} (e_r · e_p)
        """
        jd_reqs = self.chunker.split_jd_requirements(jd_text)
        cv_chunks = self.chunker.chunk_cv(cv_text)

        # Q có shape (n_req, 384), P có shape (n_chunk, 384)
        q_embs = self.encode_sentences(jd_reqs)
        p_embs = self.encode_sentences(cv_chunks)

        # Dot-product ma trận: S = Q x P^T có shape (n_req, n_chunk)
        sim_matrix = np.dot(q_embs, p_embs.T)

        # Lấy max theo chiều CV chunk, sau đó lấy trung bình trên tất cả các yêu cầu JD
        max_per_req = np.max(sim_matrix, axis=1)
        score = float(np.mean(max_per_req))
        return score

    def compute_pair_similarity_truncated(self, jd_text: str, cv_text: str) -> float:
        """Tính điểm tương đồng khi cắt cụt 256 tokens (E3a)."""
        embs = self.encode_sentences([jd_text, cv_text])
        score = float(np.dot(embs[0], embs[1]))
        return score

    def compute_batch_similarities(
        self,
        jd_texts: List[str],
        cv_texts: List[str],
        use_chunking: bool = True,
        desc: str = "Đang tính độ tương đồng ngữ nghĩa",
    ) -> np.ndarray:
        """Tính độ tương đồng cho một danh sách các cặp (JD, CV)."""
        scores = []
        compute_fn = (
            self.compute_pair_similarity_chunked
            if use_chunking
            else self.compute_pair_similarity_truncated
        )

        for jd, cv in tqdm(zip(jd_texts, cv_texts), total=len(jd_texts), desc=desc):
            s = compute_fn(jd, cv)
            scores.append(s)

        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def compute_percentiles(
        train_scores: np.ndarray,
        p_low: float = 5.0,
        p_high: float = 95.0,
    ) -> Tuple[float, float]:
        """Tìm phân vị p_low% (a) và p_high% (b) trên tập Train."""
        a = float(np.percentile(train_scores, p_low))
        b = float(np.percentile(train_scores, p_high))
        return a, b

    @staticmethod
    def scale_to_100(
        scores: Union[float, np.ndarray],
        a: float,
        b: float,
    ) -> Union[float, np.ndarray]:
        """Chuẩn hóa điểm ngữ nghĩa về thang [0, 100]: S_sem = 100 * clip((sim - a) / (b - a), 0, 1)."""
        if b <= a:
            return 50.0 if isinstance(scores, float) else np.full_like(scores, 50.0)

        normalized = (scores - a) / (b - a)
        clipped = np.clip(normalized, 0.0, 1.0) * 100.0
        return float(clipped) if isinstance(scores, float) else clipped

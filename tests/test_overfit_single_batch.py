# Level 1: Thư viện chuẩn Python (Standard Library)
import os
import sys

# Level 2: Thư viện bên thứ ba (Third-party Packages)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Level 3: Module nội bộ trong dự án (Project-Local Imports)
from src.data.load import load_split
from src.utils.metrics import calculate_macro_f1, calculate_qwk

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3


class MatchClassifierHead(nn.Module):
    """Classification head nhận đặc trưng kết hợp từ cặp vector nhúng (JD, CV)."""

    def __init__(self, in_dim: int = 384, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Kết hợp: jd_vec, cv_vec, khoảng cách tuyệt đối |jd - cv|, tích phần tử (jd * cv)
        self.net = nn.Sequential(
            nn.Linear(in_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, jd_vec: torch.Tensor, cv_vec: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(jd_vec - cv_vec)                                      # (B, 384)
        prod = jd_vec * cv_vec                                                 # (B, 384)
        feat = torch.cat([jd_vec, cv_vec, diff, prod], dim=1)                 # (B, 1536)
        logits = self.net(feat)                                                # (B, 3)
        return logits


def get_balanced_batch(df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Lấy 1 batch nhỏ cân bằng đủ 3 nhãn (0, 1, 2) từ tập Train."""
    samples = []
    n_per_class = batch_size // 3
    for label in [0, 1, 2]:
        sub = df[df["label"] == label].head(n_per_class)
        samples.append(sub)

    remainder = batch_size - sum(len(s) for s in samples)
    if remainder > 0:
        samples.append(df[df["label"] == 0].iloc[n_per_class : n_per_class + remainder])

    return pd.concat(samples, ignore_index=True)


def run_sanity_check() -> None:
    print(f"=== BẮT ĐẦU SANITY CHECK: OVERFIT 1 BATCH DUY NHẤT ({DEVICE.upper()}) ===")

    # 1. Nạp 1 batch mẫu từ tập train
    train_df = load_split("train")
    batch_df = get_balanced_batch(train_df, batch_size=BATCH_SIZE)
    print(f"Số lượng mẫu: {len(batch_df)} | Phân phối nhãn: {dict(batch_df['label'].value_counts())}")

    # 2. Trích xuất vector nhúng tĩnh qua MiniLM (tiết kiệm VRAM cho GTX 1650 4GB)
    print("Trích xuất vector nhúng từ MiniLM-L12-v2...")
    st_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    with torch.no_grad():
        jd_embs = st_model.encode(
            batch_df["job_description_text"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=False,
            device=DEVICE,
        )
        cv_embs = st_model.encode(
            batch_df["resume_text"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=False,
            device=DEVICE,
        )

    labels = torch.tensor(batch_df["label"].values, dtype=torch.long, device=DEVICE)

    # 3. Khởi tạo classifier head và optimizer
    head = MatchClassifierHead(in_dim=384, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    # 4. Huấn luyện overfit trên đúng 1 batch này
    print("\nBắt đầu vòng lặp tối ưu hóa:")
    head.train()
    reached_100 = False

    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer.zero_grad()
        logits = head(jd_embs, cv_embs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_true = labels.cpu().numpy()

        macro_f1 = calculate_macro_f1(y_true, preds)
        qwk = calculate_qwk(y_true, preds)

        if epoch % 10 == 0 or macro_f1 >= 0.999:
            print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Loss: {loss.item():.5f} | Macro-F1: {macro_f1 * 100:.1f}% | QWK: {qwk:.4f}")

        if macro_f1 >= 0.999:
            reached_100 = True
            print(f"\n==================================================================")
            print(f"  THÀNH CÔNG RỰC RỠ: MACRO-F1 ĐẠT {macro_f1 * 100:.1f}% TẠI EPOCH {epoch}!")
            print(f"  Xác nhận: Pipeline toán học, Loss, Optimizer và Metrics CHUẨN XÁC 100%!")
            print(f"==================================================================")
            break

    assert reached_100, f"Thất bại: Macro-F1 chỉ đạt {macro_f1 * 100:.2f}%!"


if __name__ == "__main__":
    run_sanity_check()

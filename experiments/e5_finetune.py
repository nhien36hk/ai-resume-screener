import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
import numpy as np
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CosineSimilarityLoss
from transformers import EarlyStoppingCallback

from src.config import ARTIFACTS_DIR, DEFAULT_EMBEDDING_MODEL, FEATURES_DIR, MAX_SEQ_LENGTH, RESULTS_DIR
from src.data.loader import load_split
from src.features.text import prepare_finetune_text
from src.pipeline import load_feature_tables
from src.tuning import CRITERIA, evaluate_combined_variant, tune_combined_model
from src.utils import load_json, save_json, to_markdown_table

TARGET_MAP = {0: 0.0, 1: 0.5, 2: 1.0}


def build_pair_dataset(df: pd.DataFrame) -> Dataset:
    jd_texts = [str(t) for t in df["job_description_text"]]
    cv_texts = [prepare_finetune_text(str(t)) for t in df["resume_text"]]
    scores = [TARGET_MAP.get(int(lbl), 0.0) for lbl in df["label"]]
    return Dataset.from_dict({"sentence1": jd_texts, "sentence2": cv_texts, "score": scores})


def train_e5(train_df, val_df, output_dir):
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH

    train_dataset = build_pair_dataset(train_df)
    val_dataset = build_pair_dataset(val_df)

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=val_dataset["sentence1"],
        sentences2=val_dataset["sentence2"],
        scores=val_dataset["score"],
        name="val",
    )
    loss = CosineSimilarityLoss(model=model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="val_spearman_cosine",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    return model


def run_e5(seed: int) -> dict:
    print(f"=== CHẠY THỰC NGHIỆM E5: FINE-TUNING MINILM-L12-V2 (Seed {seed}) ===")
    df_train = load_split("train", seed)
    df_val = load_split("val", seed)
    df_test = load_split("test", seed)

    model_dir = ARTIFACTS_DIR / "models" / "e5_finetuned" / f"seed{seed}"
    if (model_dir / "config.json").exists():
        print(f"Nạp mô hình đã fine-tune có sẵn từ {model_dir}...")
        model = SentenceTransformer(str(model_dir))
    else:
        print(f"Bắt đầu huấn luyện Fine-tune mô hình MiniLM cho seed {seed}...")
        model = train_e5(df_train, df_val, model_dir)

    feat_dir = FEATURES_DIR / "minilm_finetuned" / f"seed{seed}"
    tables = load_feature_tables(feat_dir, model, seed)
    percentile_p5, percentile_p95 = (float(v) for v in np.percentile(tables["train"]["sim_chunk_200"], [5, 95]))

    print(f"\nTối ưu siêu tham số trên tập Val cho mô hình M5 (seed {seed})...")
    best_config = tune_combined_model(tables["val"], percentile_p5, percentile_p95)
    threshold = best_config.get("skill_threshold", best_config.get("tau", 0.7))
    m5_metrics, _ = evaluate_combined_variant(tables["val"], tables["test"], percentile_p5, percentile_p95, threshold, best_config["weights"])
    print(f"-> M5 tối ưu (seed {seed}): skill_threshold={threshold}, t=({best_config['t1']:.0f}, {best_config['t2']:.0f})")
    best_config = {**best_config, "weights": dict(zip(CRITERIA, best_config["weights"].tolist())), "scaler": (percentile_p5, percentile_p95)}

    e4_path = RESULTS_DIR / f"seed{seed}" / "e4.json"
    e4_data = load_json(e4_path, {})
    m4_metrics = next((r for r in e4_data.get("results", []) if "Full" in r.get("variant", r.get("name", ""))), None)

    headers = ["Mô hình", "QWK", "Δ QWK", "Macro-F1", "Δ F1", "MAE", "Spearman", "NDCG@10"]
    rows = []

    if m4_metrics:
        rows.append([
            f"M4: Pretrained MiniLM (seed {seed})",
            f"{m4_metrics['qwk']:.4f}",
            "-",
            f"{m4_metrics['macro_f1']:.4f}",
            "-",
            f"{m4_metrics['mae']:.4f}",
            f"{m4_metrics['spearman']:.4f}",
            f"{m4_metrics['ndcg_10']:.4f}",
        ])
        d_qwk = m5_metrics["qwk"] - m4_metrics["qwk"]
        d_f1 = m5_metrics["macro_f1"] - m4_metrics["macro_f1"]
        d_qwk_str = f"{d_qwk:+.4f}"
        d_f1_str = f"{d_f1:+.4f}"
    else:
        d_qwk_str = "-"
        d_f1_str = "-"

    rows.append([
        f"M5: Fine-tuned MiniLM (seed {seed})",
        f"{m5_metrics['qwk']:.4f}",
        d_qwk_str,
        f"{m5_metrics['macro_f1']:.4f}",
        d_f1_str,
        f"{m5_metrics['mae']:.4f}",
        f"{m5_metrics.get('spearman', 0.0):.4f}",
        f"{m5_metrics.get('ndcg_10', 0.0):.4f}",
    ])

    md_table = to_markdown_table(headers, rows)
    print("\n" + md_table + "\n")

    out_dir = RESULTS_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"seed": seed, "config": best_config, "metrics": m5_metrics}
    save_json(out_dir / "e5.json", res)
    with open(out_dir / "e5.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả Thực nghiệm E5: So sánh M4 Pretrained vs M5 Fine-tuned (Seed {seed})\n\n" + md_table + "\n")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed phân chia dữ liệu.")
    args = parser.parse_args()
    run_e5(args.seed)

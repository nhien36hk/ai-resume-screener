"""Vẽ 4 hình minh họa cho Chương 4 luận văn (PNG 300 DPI) và kiểm tra khớp số liệu."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from src.config import ARTIFACTS_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_SEEDS, MAX_SEQ_LENGTH, RESULTS_DIR
from src.data.loader import load_all
from src.utils import load_json

FIGURES_DIR = ARTIFACTS_DIR / "figures"
DPI = 300
SEEDS = list(DEFAULT_SEEDS)
LABELS_ORDER = ["No Fit (0)", "Potential Fit (1)", "Good Fit (2)"]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def fmt_vi(val: float, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{int(round(val)):,}".replace(",", ".")
    return f"{val:.{decimals}f}".replace(".", ",")


def plot_fig_4_1(out_dir: Path) -> Path:
    df = load_all()
    stats = load_json(RESULTS_DIR / "data_stats.json")["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EMBEDDING_MODEL)

    counts = [int((df["label"] == i).sum()) for i in (0, 1, 2)]
    assert counts == [stats["labels"]["No Fit"], stats["labels"]["Potential Fit"], stats["labels"]["Good Fit"]]
    assert sum(counts) == 8000

    cvs = df.drop_duplicates("resume_text")["resume_text"].tolist()
    jds = df.drop_duplicates("job_description_text")["job_description_text"].tolist()
    cv_tokens = np.array([len(tokenizer.encode(t, add_special_tokens=False)) for t in cvs])
    jd_tokens = np.array([len(tokenizer.encode(t, add_special_tokens=False)) for t in jds])
    assert len(cv_tokens) == stats["n_cv_texts"] and len(jd_tokens) == stats["n_jd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax1.bar(LABELS_ORDER, counts, color=colors, width=0.55)
    ax1.set_title("Phân bố nhãn (8.000 cặp CV–JD)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số cặp")
    ax1.set_ylim(0, 4800)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, cnt in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 100,
            f"{fmt_vi(cnt)}\n({fmt_vi(cnt / 80, 1)}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bp = ax2.boxplot(
        [cv_tokens, jd_tokens],
        tick_labels=[f"CV ({len(cv_tokens)})", f"JD ({len(jd_tokens)})"],
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
        medianprops={"color": "#111111", "linewidth": 2},
    )
    for patch, col in zip(bp["boxes"], ["#4C72B0", "#55A868"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax2.axhline(MAX_SEQ_LENGTH, color="#C44E52", linestyle="--", linewidth=1.6, label=f"Giới hạn {MAX_SEQ_LENGTH} token")
    ax2.set_title("Phân bố độ dài văn bản theo token (MiniLM)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Số token")
    ax2.set_ylim(0, max(int(cv_tokens.max()), int(jd_tokens.max())) * 1.08)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)

    cv_over = (cv_tokens > MAX_SEQ_LENGTH).mean() * 100
    jd_over = (jd_tokens > MAX_SEQ_LENGTH).mean() * 100
    ax2.text(
        0.58,
        0.72,
        f"CV: TB {fmt_vi(cv_tokens.mean())}, TV {fmt_vi(np.median(cv_tokens))}, >256: {fmt_vi(cv_over, 1)}%\n"
        f"JD: TB {fmt_vi(jd_tokens.mean())}, TV {fmt_vi(np.median(jd_tokens))}, >256: {fmt_vi(jd_over, 1)}%",
        transform=ax2.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8F9FA", "edgecolor": "#CCCCCC"},
    )

    fig.suptitle("Hình 4.1. Đặc điểm phân bố nhãn và độ dài token trong bộ dữ liệu", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig_4_1_data_distribution.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig_4_2(out_dir: Path) -> Path:
    tables = load_json(RESULTS_DIR / "thesis_tables.json")
    t45, t46, t48 = tables["table_4_5"], tables["table_4_6"], tables["table_4_8"]
    assert abs(t46["M4"]["macro_f1"]["mean"] - t48["M4"]["macro_f1"]["mean"]) < 1e-9
    assert abs(t46["M4"]["qwk"]["mean"] - t48["M4"]["qwk"]["mean"]) < 1e-9

    specs = [
        ("M1\n(Từ khóa)", t45["M1"]),
        ("M2\n(TF-IDF)", t45["M2"]),
        ("M3a\n(Cắt 256)", t45["M3a"]),
        ("M3b\n(Chia đoạn)", t45["M3b"]),
        ("M4\n(Kết hợp)", t46["M4"]),
        ("M5\n(Tinh chỉnh)", t48["M5"]),
    ]
    names = [s[0] for s in specs]
    f1_means = [s[1]["macro_f1"]["mean"] for s in specs]
    f1_sds = [s[1]["macro_f1"]["sd"] for s in specs]
    qwk_means = [s[1]["qwk"]["mean"] for s in specs]
    qwk_sds = [s[1]["qwk"]["sd"] for s in specs]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    b1 = ax.bar(x - w / 2, f1_means, w, yerr=f1_sds, capsize=4, label="Macro-F1", color="#4C72B0", alpha=0.9)
    b2 = ax.bar(x + w / 2, qwk_means, w, yerr=qwk_sds, capsize=4, label="QWK", color="#DD8452", alpha=0.9)

    for bar, mean, sd in zip(b1, f1_means, f1_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.012, fmt_vi(mean, 3), ha="center", va="bottom", fontsize=8.5)
    for bar, mean, sd in zip(b2, qwk_means, qwk_sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.012, fmt_vi(mean, 3), ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("Giá trị chỉ số (Mean ± SD, 5 seed)")
    ax.set_ylim(0, 0.54)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("Hình 4.2. So sánh Macro-F1 và QWK của năm phương pháp M1–M5 (chia theo JD)", fontsize=11.5, fontweight="bold")

    fig.tight_layout()
    out_path = out_dir / "fig_4_2_model_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig_4_3(out_dir: Path) -> Path:
    mats = {"M2 (TF-IDF)": np.zeros((3, 3), dtype=int), "M4 (Kết hợp)": np.zeros((3, 3), dtype=int), "M5 (Tinh chỉnh)": np.zeros((3, 3), dtype=int)}
    for seed in SEEDS:
        s_dir = RESULTS_DIR / "jd" / f"seed{seed}"
        e2 = load_json(s_dir / "e2.json")
        e4 = load_json(s_dir / "e4.json")
        e5 = load_json(s_dir / "e5.json")
        mats["M2 (TF-IDF)"] += np.asarray(e2["metrics"]["confusion"], dtype=int)
        mats["M4 (Kết hợp)"] += np.asarray(next(r for r in e4["results"] if r["code"] == "M4")["confusion"], dtype=int)
        mats["M5 (Tinh chỉnh)"] += np.asarray(next(r for r in e5["results"] if r["code"] == "M5")["confusion"], dtype=int)

    row_sums = None
    for name, mat in mats.items():
        assert int(mat.sum()) == 6029, f"{name} tổng != 6029"
        cur_rows = mat.sum(axis=1).tolist()
        row_sums = cur_rows if row_sums is None else row_sums
        assert cur_rows == row_sums, f"{name} tổng từng hàng lệch giữa các mô hình"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    short_labels = ["No Fit\n(0)", "Potential\n(1)", "Good Fit\n(2)"]
    for ax, (title, mat) in zip(axes, mats.items()):
        row_pct = mat / mat.sum(axis=1, keepdims=True) * 100.0
        ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=75)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_xlabel("Nhãn dự đoán", fontsize=9.5)
        if ax is axes[0]:
            ax.set_ylabel("Nhãn thật", fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        for i in range(3):
            for j in range(3):
                val, pct = mat[i, j], row_pct[i, j]
                color = "white" if pct > 42 else "black"
                ax.text(j, i, f"{fmt_vi(val)}\n({fmt_vi(pct, 1)}%)", ha="center", va="center", color=color, fontsize=9)

    fig.suptitle("Hình 4.3. Ma trận nhầm lẫn cộng dồn 5 seed (6.029 cặp kiểm tra, chia theo JD)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "fig_4_3_confusion_matrices.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig_4_4(out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for seed, color in zip(SEEDS, palette):
        tl = load_json(ARTIFACTS_DIR / "models" / "e5_finetuned" / "jd" / f"seed{seed}" / "train_log.json")
        hist = tl["log_history"]
        train_pts = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
        eval_pts = [(h["epoch"], h["eval_val_spearman_cosine"]) for h in hist if "eval_val_spearman_cosine" in h]
        best_ep, best_sp = max(eval_pts, key=lambda x: x[1])
        stop_ep = int(round(eval_pts[-1][0]))

        ax1.plot([p[0] for p in train_pts], [p[1] for p in train_pts], color=color, linewidth=1.6, label=f"Seed {seed} ({tl['train_seconds']:.0f}s)")
        ax2.plot([p[0] for p in eval_pts], [p[1] for p in eval_pts], marker="o", color=color, linewidth=1.7,
                 label=f"Seed {seed} (best ep {int(round(best_ep))}, dừng ep {stop_ep})")
        ax2.scatter([best_ep], [best_sp], marker="*", s=140, color=color, edgecolor="black", zorder=5)

    ax1.set_title("Mất mát huấn luyện theo epoch (CosineSimilarityLoss)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.5)

    ax2.set_title("Spearman trên tập kiểm định theo epoch (dừng sớm patience=2)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Spearman (cosine)")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_ylim(0.14, 0.58)
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8)

    fig.suptitle("Hình 4.4. Đường mất mát và Spearman kiểm định khi tinh chỉnh M5 (5 seed, chia theo JD)", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "fig_4_4_m5_training.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Vẽ 4 hình minh họa Chương 4.")
    parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_fig_4_1(FIGURES_DIR),
        plot_fig_4_2(FIGURES_DIR),
        plot_fig_4_3(FIGURES_DIR),
        plot_fig_4_4(FIGURES_DIR),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 10_000, f"File hình không hợp lệ: {p}"
        print(f"Đã lưu: {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

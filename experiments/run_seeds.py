"""Chạy thực nghiệm qua các seed và tổng hợp báo cáo trung bình ± độ lệch chuẩn."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import DEFAULT_SEEDS, RESULTS_DIR
from src.utils import load_json, to_markdown_table

from experiments.e1_keyword import run_e1
from experiments.e2_tfidf import run_e2
from experiments.e3_st_pretrained import run_e3
from experiments.e4_combined_ablation import run_e4
from experiments.e5_finetune import run_e5

EXP_RUNNERS = {
    "e1": run_e1,
    "e2": run_e2,
    "e3": run_e3,
    "e4": run_e4,
    "e5": run_e5,
}

METRIC_KEYS = ["qwk", "macro_f1", "mae", "spearman", "ndcg_10"]


def format_stat(vals: list[float]) -> str:
    if not vals:
        return "-"
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{mean:.4f} ± {sd:.4f}"


def build_summary(seeds: list[int]) -> str:
    collected = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        s_dir = RESULTS_DIR / f"seed{seed}"
        e1 = load_json(s_dir / "e1.json")
        if e1:
            for k in METRIC_KEYS:
                collected["E1: Baseline Keyword"][k].append(e1["metrics"].get(k, 0.0))

        e2 = load_json(s_dir / "e2.json")
        if e2:
            for k in METRIC_KEYS:
                collected["E2: Baseline TF-IDF"][k].append(e2["metrics"].get(k, 0.0))

        e3 = load_json(s_dir / "e3.json")
        if e3:
            for k in METRIC_KEYS:
                collected["E3a: Truncate 256"][k].append(e3["e3a"]["metrics"].get(k, 0.0))
                collected["E3b: Chunking 200/50"][k].append(e3["e3b"]["metrics"].get(k, 0.0))

        e4 = load_json(s_dir / "e4.json")
        if e4:
            for r in e4.get("results", []):
                var = r.get("variant", r.get("name", ""))
                # Bỏ hậu tố seed nếu có
                var_clean = var.split(" (seed")[0]
                for k in METRIC_KEYS:
                    collected[var_clean][k].append(r.get(k, 0.0))

        e5 = load_json(s_dir / "e5.json")
        if e5:
            if "results" in e5:
                for r in e5["results"]:
                    var = r.get("variant", r.get("name", ""))
                    var_clean = var.split(" (seed")[0]
                    for k in METRIC_KEYS:
                        collected[var_clean][k].append(r.get(k, 0.0))
            else:
                for k in METRIC_KEYS:
                    collected["M5: Fine-tuned MiniLM"][k].append(e5["metrics"].get(k, 0.0))

    headers = ["Mô hình / Biến thể", "Số seed", "QWK", "Macro-F1", "MAE", "Spearman", "NDCG@10"]
    rows = []
    for model_name, metrics in collected.items():
        n_seeds = len(metrics["qwk"])
        rows.append([
            model_name,
            str(n_seeds),
            format_stat(metrics["qwk"]),
            format_stat(metrics["macro_f1"]),
            format_stat(metrics["mae"]),
            format_stat(metrics["spearman"]),
            format_stat(metrics["ndcg_10"]),
        ])

    md_table = to_markdown_table(headers, rows)
    out_file = RESULTS_DIR / "summary.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Tổng hợp kết quả thực nghiệm qua các Seeds (Mean ± SD)\n\n" + md_table + "\n")
    return md_table


def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm trên nhiều seed và tổng hợp kết quả.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--exps", nargs="+", choices=["e1", "e2", "e3", "e4", "e5", "e6"], default=["e1", "e2", "e3", "e4", "e5"])
    args = parser.parse_args()

    active_seed_exps = [e for e in args.exps if e in EXP_RUNNERS]
    for seed in args.seeds:
        seed_dir = RESULTS_DIR / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for exp_name in active_seed_exps:
            exp_file = seed_dir / f"{exp_name}.json"
            if exp_file.exists():
                print(f"Bỏ qua {exp_name} seed {seed}: đã có {exp_file.name}")
                continue
            runner = EXP_RUNNERS[exp_name]
            runner(seed)

    if "e6" in args.exps:
        from experiments.e6_benchmark import run_e6
        run_e6()

    summary_table = build_summary(args.seeds)
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP KẾT QUẢ ĐA SEED (MEAN ± SD)")
    print("=" * 80)
    print(summary_table)
    print(f"\n-> Đã lưu bảng tổng hợp vào {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()


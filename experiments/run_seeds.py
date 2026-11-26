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

METRIC_KEYS = ["qwk", "macro_f1", "accuracy", "mae", "spearman", "spearman_jd", "ndcg_10"]
CONFUSION_MODELS = {"M2": "E2: Baseline TF-IDF", "M4": "M4: Full Model", "M5": "M5: Full Model"}


def format_stat(vals: list[float]) -> str:
    if not vals:
        return "-"
    mean = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{mean:.4f} ± {sd:.4f}"


def build_summary(seeds: list[int], scheme: str = "jd") -> str:
    collected = defaultdict(lambda: defaultdict(list))
    confusion = {}  # mã mô hình -> ma trận nhầm lẫn cộng dồn qua các seed (hàng = nhãn thật)

    def add(name, metrics, code=None):
        for k in METRIC_KEYS:
            collected[name][k].append(metrics.get(k, np.nan))
        if code in CONFUSION_MODELS and metrics.get("confusion") is not None:
            confusion[code] = confusion.get(code, 0) + np.asarray(metrics["confusion"])

    for seed in seeds:
        s_dir = RESULTS_DIR / scheme / f"seed{seed}"
        e1 = load_json(s_dir / "e1.json")
        if e1:
            add("E1: Baseline Keyword", e1["metrics"])
        e2 = load_json(s_dir / "e2.json")
        if e2:
            add("E2: Baseline TF-IDF", e2["metrics"], "M2")
        e3 = load_json(s_dir / "e3.json")
        if e3:
            add("E3a: Truncate 256", e3["e3a"]["metrics"])
            add("E3b: Chunking 200/50", e3["e3b"]["metrics"])
        for exp in ("e4", "e5"):
            data = load_json(s_dir / f"{exp}.json")
            for r in (data or {}).get("results", []):
                add(r.get("variant", r.get("name", "")).split(" (seed")[0], r, r.get("code"))

    headers = ["Mô hình / Biến thể", "Số seed", "QWK", "Macro-F1", "Accuracy", "MAE", "Spearman", "Spearman/JD", "NDCG@10"]
    rows = [[name, str(len(m["qwk"]))] + [format_stat([v for v in m[k] if not np.isnan(v)]) for k in METRIC_KEYS]
            for name, m in collected.items()]
    md_table = to_markdown_table(headers, rows)

    parts = [f"# Tổng hợp kết quả thực nghiệm qua các Seeds (Mean ± SD, chia {scheme})\n\n" + md_table + "\n"]
    for code, mat in confusion.items():
        parts.append(f"\n## Ma trận nhầm lẫn cộng dồn {len(seeds)} seed: {CONFUSION_MODELS[code]} ({code})\n\n"
                     + to_markdown_table(["Nhãn thật \\ Dự đoán", "0", "1", "2"],
                                         [[str(i)] + [str(int(v)) for v in row] for i, row in enumerate(mat)]) + "\n")
    out_file = RESULTS_DIR / scheme / "summary.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm trên nhiều seed và tổng hợp kết quả.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--exps", nargs="+", choices=["e1", "e2", "e3", "e4", "e5", "e6"], default=["e1", "e2", "e3", "e4", "e5"])
    parser.add_argument("--force", action="store_true", help="Chạy lại và ghi đè kết quả đã có thay vì bỏ qua.")
    parser.add_argument("--scheme", choices=["jd", "strict"], default="jd", help="Cách chia dữ liệu.")
    args = parser.parse_args()

    active_seed_exps = [e for e in args.exps if e in EXP_RUNNERS]
    for seed in args.seeds:
        seed_dir = RESULTS_DIR / args.scheme / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for exp_name in active_seed_exps:
            exp_file = seed_dir / f"{exp_name}.json"
            if not args.force and exp_file.exists():
                print(f"Bỏ qua {exp_name} seed {seed}: đã có {exp_file.name}")
                continue
            runner = EXP_RUNNERS[exp_name]
            runner(seed, args.scheme)

    if "e6" in args.exps:
        e6_file = RESULTS_DIR / "e6.json"
        if not args.force and e6_file.exists():
            print(f"Bỏ qua e6: đã có {e6_file.name}")
        else:
            from experiments.e6_benchmark import run_e6
            run_e6()

    summary_table = build_summary(args.seeds, args.scheme)
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP KẾT QUẢ ĐA SEED (MEAN ± SD)")
    print("=" * 80)
    print(summary_table)
    print(f"\n-> Đã lưu bảng tổng hợp vào {RESULTS_DIR / args.scheme / 'summary.md'}")


if __name__ == "__main__":
    main()


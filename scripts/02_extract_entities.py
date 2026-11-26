"""Extract JD/CV entities into the JSON caches.

Re-running only processes items that are missing, outdated (older schema version) or invalid.
    python scripts/02_extract_entities.py --dry-run
    python scripts/02_extract_entities.py --backend vllm --model Qwen/Qwen3-14B-FP8
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CV_CACHE_PATH,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_VLLM_MODEL,
    DEGREE_LEVELS,
    FAILED_EXTRACTIONS_PATH,
    JD_CACHE_PATH,
    MAJOR_GROUPS,
    RAW_DIR,
)
from src.data.extractor import NvidiaAPIBackend, VLLMBackend, extract_many
from src.data.schemas import CVEntities, JDEntities, is_current
from src.utils import compute_text_hash, load_json, save_json, to_markdown_table

KINDS = {"jd": JDEntities, "cv": CVEntities}


def collect_items(raw_path: Path = RAW_DIR / "raw_dataset.parquet") -> dict[str, dict[str, str]]:
    """Unique documents: JDs keyed by job_id, CVs by text hash."""
    df = pd.read_parquet(raw_path)
    return {
        "jd": dict(zip(df["job_id"].astype(str), df["job_description_text"])),
        "cv": {compute_text_hash(t): t for t in df["resume_text"].unique()},
    }


def plan(kind: str, items: dict[str, str], cache: dict, failed: dict, args: argparse.Namespace) -> list[str]:
    if args.retry_failed:
        keys = [k for k in items if k in failed]
    elif args.force:
        keys = list(items)
    else:
        keys = [k for k in items if not is_current(KINDS[kind], cache.get(k))]
    if args.sample:
        keys = sorted(keys, key=lambda k: k not in failed)[:args.sample]  # previously failed first, to test the fix
    return keys


def run_task(kind: str, keys: list[str], items: dict[str, str], cache: dict, failed: dict,
             backend, chunk_size: int, checkpoint) -> tuple[int, int]:
    """Extract in chunks; each chunk is one batched backend call, saved before the next one starts."""
    ok = bad = 0
    for start in tqdm(range(0, len(keys), chunk_size), desc=f"{kind.upper()} chunks"):
        chunk = keys[start:start + chunk_size]
        for key, result in zip(chunk, extract_many(kind, [items[k] for k in chunk], backend)):
            if isinstance(result, dict):
                cache[key] = result
                failed.pop(key, None)
                ok += 1
            else:
                failed[key] = {"error": result, "timestamp": time.time(), "preview": items[key][:300]}
                bad += 1
        checkpoint()
    return ok, bad


def backup(path: Path) -> None:
    target = path.with_suffix(".bak.json")
    if path.exists() and not target.exists():
        shutil.copy2(path, target)


def print_distribution(title: str, values: list[str], jd: Counter, cv: Counter) -> None:
    rows = [[v, jd[v], cv[v]] for v in values if jd[v] or cv[v]]
    print(f"\n{title}:\n" + to_markdown_table(["Giá trị", "JD", "CV"], rows))


def print_summary(items: dict, caches: dict, counts: dict, elapsed: float) -> None:
    current = {k: [caches[k][key] for key in items[k] if is_current(m, caches[k].get(key))] for k, m in KINDS.items()}
    rows = [[k.upper(), len(items[k]), *counts.get(k, (0, 0)), len(current[k])] for k in KINDS]
    print("\n" + to_markdown_table(["Loại", "Tổng", "Mới trích", "Lỗi", "Hợp lệ"], rows))
    print(f"Thời gian: {elapsed:.1f}s")
    print_distribution("Nhóm ngành", MAJOR_GROUPS,
                       Counter(e["major_group"] for e in current["jd"]), Counter(e["major_group"] for e in current["cv"]))
    print_distribution("Bậc học", DEGREE_LEVELS,
                       Counter(e["min_degree"] for e in current["jd"]), Counter(e["highest_degree"] for e in current["cv"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trích xuất thực thể JD/CV bằng LLM (NVIDIA API hoặc vLLM).")
    p.add_argument("--raw-path", type=Path, default=RAW_DIR / "raw_dataset.parquet")
    p.add_argument("--jd-cache", type=Path, default=JD_CACHE_PATH)
    p.add_argument("--cv-cache", type=Path, default=CV_CACHE_PATH)
    p.add_argument("--failed-file", type=Path, default=FAILED_EXTRACTIONS_PATH)
    p.add_argument("--backend", choices=["nvidia", "vllm"], default="nvidia")
    p.add_argument("--model", help="Mặc định theo backend.")
    p.add_argument("--chunk-size", type=int, help="Số văn bản mỗi lần gọi model (mặc định: vllm 256, nvidia 20).")
    p.add_argument("--workers", type=int, default=4, help="Số luồng gọi API song song.")
    p.add_argument("--rpm", type=float, default=35.0, help="Giới hạn request/phút của API.")
    p.add_argument("--sample", type=int, help="Chỉ xử lý N mục mỗi loại (ưu tiên mục từng lỗi).")
    p.add_argument("--only", choices=["jd", "cv", "all"], default="all")
    p.add_argument("--retry-failed", action="store_true", help="Chỉ chạy lại mục trong failed_extractions.json.")
    p.add_argument("--force", action="store_true", help="Trích xuất lại tất cả.")
    p.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch, không gọi model.")
    args = p.parse_args()
    vllm = args.backend == "vllm"
    args.model = args.model or (DEFAULT_VLLM_MODEL if vllm else DEFAULT_NVIDIA_MODEL)
    args.chunk_size = args.chunk_size or (256 if vllm else 20)
    return args


def main() -> None:
    args = parse_args()
    items = collect_items(args.raw_path)
    caches = {"jd": load_json(args.jd_cache, {}), "cv": load_json(args.cv_cache, {})}
    failed_all = load_json(args.failed_file, {})
    failed = {k: failed_all.setdefault(f"{k}s", {}) for k in KINDS}

    kinds = list(KINDS) if args.only == "all" else [args.only]
    todo = {k: plan(k, items[k], caches[k], failed[k], args) for k in kinds}
    rows = [[k.upper(), len(items[k]), len(todo[k]), len(failed[k])] for k in kinds]
    print(f"Backend: {args.backend} | Model: {args.model} | Chunk: {args.chunk_size}")
    print(to_markdown_table(["Loại", "Tổng", "Cần trích", "Từng lỗi"], rows))
    if args.dry_run or not any(todo.values()):
        return

    backup(args.jd_cache)
    backup(args.cv_cache)
    backend = (VLLMBackend(args.model) if args.backend == "vllm"
               else NvidiaAPIBackend(args.model, workers=args.workers, rpm=args.rpm))

    def checkpoint() -> None:
        save_json(args.jd_cache, caches["jd"])
        save_json(args.cv_cache, caches["cv"])
        save_json(args.failed_file, failed_all)

    start, counts = time.time(), {}
    try:
        for k in kinds:
            counts[k] = run_task(k, todo[k], items[k], caches[k], failed[k], backend, args.chunk_size, checkpoint)
    except KeyboardInterrupt:
        checkpoint()
        print("\nĐã dừng. Các chunk xong đã được lưu, chạy lại để tiếp tục.")
        return
    print_summary(items, caches, counts, time.time() - start)


if __name__ == "__main__":
    main()

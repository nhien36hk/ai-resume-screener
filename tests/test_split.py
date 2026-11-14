from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_split


def test_split_files_load_and_columns():
    for name in ["train", "val", "test"]:
        df = load_split(name)
        assert len(df) > 0, f"Split {name} rỗng!"
        expected_cols = [
            "pair_id",
            "job_id",
            "label",
            "label_text",
            "job_description_text",
            "resume_text",
            "jd_entities",
            "cv_entities",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Thiếu cột {col} trong split {name}"

        # Kiểm tra nhãn chuẩn
        valid_labels = {0, 1, 2}
        unique_labels = set(df["label"].unique())
        assert unique_labels.issubset(valid_labels), f"Nhãn không hợp lệ trong {name}: {unique_labels}"

        # Kiểm tra entities không bị null/None
        sample = df.iloc[0]
        assert isinstance(sample.jd_entities, dict), "jd_entities phải là dict"
        assert isinstance(sample.cv_entities, dict), "cv_entities phải là dict"


def test_zero_leakage_between_splits():
    df_train = load_split("train")
    df_val = load_split("val")
    df_test = load_split("test")

    train_jds = set(df_train["job_id"].unique())
    val_jds = set(df_val["job_id"].unique())
    test_jds = set(df_test["job_id"].unique())

    overlap_train_val = train_jds.intersection(val_jds)
    overlap_train_test = train_jds.intersection(test_jds)
    overlap_val_test = val_jds.intersection(test_jds)

    assert len(overlap_train_val) == 0, f"Rò rỉ JD giữa Train và Val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Rò rỉ JD giữa Train và Test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Rò rỉ JD giữa Val và Test: {overlap_val_test}"


if __name__ == "__main__":
    print("Đang chạy kiểm thử tính toàn vẹn các tập dữ liệu...")
    test_split_files_load_and_columns()
    print("✅ PASS: test_split_files_load_and_columns (Cấu trúc cột & kiểu dữ liệu chuẩn)")
    test_zero_leakage_between_splits()
    print("✅ PASS: test_zero_leakage_between_splits (0 job_id trùng lặp giữa Train, Val, Test)")


"""Làm sạch dữ liệu Palmer Penguins theo quyết định trong data/data_quality.md.

Cách chạy (từ thư mục gốc của project):
    python ml/src/clean_data.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_CSV = DATA_DIR / "raw" / "penguins.csv"
CLEAN_CSV = DATA_DIR / "processed" / "penguins_clean.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Bỏ row trùng lặp (hiện tại không có, nhưng vẫn giữ để an toàn nếu dữ liệu thay đổi)
    df = df.drop_duplicates()

    # Bỏ mọi row có giá trị thiếu: 2 row thiếu toàn bộ số đo + 9 row chỉ thiếu sex
    df = df.dropna()

    # Sau khi bỏ NA, hai cột này trở lại số nguyên đúng như data dictionary
    df = df.astype({"flipper_length_mm": "int64", "body_mass_g": "int64"})

    return df.reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    cleaned = clean(raw)

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)

    print(f"Đọc: {RAW_CSV} ({len(raw)} row)")
    print(f"Ghi: {CLEAN_CSV} ({len(cleaned)} row, bỏ {len(raw) - len(cleaned)} row)")
    print("\nSpecies sau khi làm sạch:")
    for species, count in cleaned["species"].value_counts().items():
        print(f"  {species}: {count}")


if __name__ == "__main__":
    main()

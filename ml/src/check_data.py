"""Kiểm tra chất lượng dữ liệu Palmer Penguins.

Cách chạy (từ thư mục gốc của project):
    python ml/src/check_data.py
    python ml/src/check_data.py duong/dan/khac.csv
"""

import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "penguins.csv"


def print_section(title: str) -> None:
    print(f"\n{title}:")


def check_data(csv_path: Path) -> None:
    # pandas mặc định hiểu chuỗi "NA" trong CSV là giá trị thiếu (NaN)
    df = pd.read_csv(csv_path)

    print(f"File: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.shape[1]}")

    print_section("Kiểu dữ liệu")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    print_section("Missing")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("  Không có giá trị thiếu")
    for col, count in missing.items():
        print(f"  {col}: {count}")

    rows_with_null = df[df.isna().any(axis=1)]
    print(f"  => Số row có ít nhất 1 giá trị thiếu: {len(rows_with_null)}")
    if not rows_with_null.empty:
        print(rows_with_null.to_string())

    print_section("Duplicate")
    duplicated = df[df.duplicated(keep=False)]
    print(f"  Số row trùng lặp hoàn toàn: {df.duplicated().sum()}")
    if not duplicated.empty:
        print(duplicated.to_string())

    print_section("Species")
    for species, count in df["species"].value_counts(dropna=False).items():
        print(f"  {species}: {count}")

    print_section("Min/Max các cột số")
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        print(f"  {col}: min={numeric[col].min()}, max={numeric[col].max()}")

    print_section("Giá trị của các cột phân loại")
    for col in df.select_dtypes(exclude="number").columns:
        values = df[col].value_counts(dropna=False).to_dict()
        print(f"  {col}: {values}")

    print_section("Nếu bỏ mọi row có giá trị thiếu")
    cleaned = df.dropna()
    print(f"  Rows còn lại: {len(cleaned)} (mất {len(df) - len(cleaned)} row)")
    for species, count in cleaned["species"].value_counts().items():
        print(f"  {species}: {count}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    check_data(path)

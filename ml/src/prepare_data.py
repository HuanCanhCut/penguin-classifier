"""Chia dữ liệu sạch thành tập train (80%) và test (20%).

Cách chạy (từ thư mục gốc của project, sau khi chạy clean_data.py):
    python ml/src/prepare_data.py

Quy tắc từ bước này trở đi:
- train.csv: dùng để phát triển model (chọn đặc trưng, tiền xử lý, chọn model, cross-validation).
- test.csv: giữ kín tới cuối, chỉ dùng MỘT lần để đánh giá model cuối cùng.
  Không mở ra xem, không dùng để chỉnh model, không fit scaler/encoder trên nó.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "penguins_clean.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

TARGET = "species"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)

    # Chỉ chia, không tiền xử lý gì ở đây: scaler/encoder sẽ được fit trên train ở bước huấn luyện
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Ghi lại cả đặc trưng lẫn nhãn vào cùng một file, giữ nguyên thứ tự cột như dữ liệu gốc
    train = pd.concat([X_train, y_train], axis=1)[df.columns]
    test = pd.concat([X_test, y_test], axis=1)[df.columns]
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    print(f"Đọc: {CLEAN_CSV} ({len(df)} row)")
    print(f"Ghi: {TRAIN_CSV} ({len(train)} row)")
    print(f"Ghi: {TEST_CSV} ({len(test)} row)")

    # Chỉ in số lượng mỗi loài để kiểm tra stratify, không in nội dung tập test
    print("\nTỉ lệ mỗi loài (toàn bộ / train / test):")
    for species in y.value_counts().index:
        print(
            f"  {species}: "
            f"{(y == species).mean():.1%} / "
            f"{(y_train == species).mean():.1%} / "
            f"{(y_test == species).mean():.1%}"
        )


if __name__ == "__main__":
    main()

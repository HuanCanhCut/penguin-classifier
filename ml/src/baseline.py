"""Baseline: luôn đoán loài phổ biến nhất trong tập train.

Cách chạy (từ thư mục gốc của project, sau khi chạy prepare_data.py):
    python ml/src/baseline.py

Đánh giá bằng cross-validation 5 fold trên tập train, không dùng test.
Tập test chỉ dùng một lần ở cuối để so sánh model cuối cùng với baseline.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

TRAIN_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "train.csv"

TARGET = "species"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
RANDOM_STATE = 42


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    # DummyClassifier không nhìn vào đặc trưng, nhưng vẫn truyền X để cùng giao diện với các model sau
    X = train[FEATURES]
    y = train[TARGET]

    model = DummyClassifier(strategy="most_frequent")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])

    model.fit(X, y)
    print(f"Train: {len(train)} row, loài phổ biến nhất: {model.classes_[model.class_prior_.argmax()]}")
    print("Đánh giá: cross-validation 5 fold (stratified) trên tập train\n")

    accuracy = scores["test_accuracy"]
    f1_macro = scores["test_f1_macro"]
    print(f"Baseline Accuracy: {accuracy.mean():.2f} (± {accuracy.std():.2f})")
    print(f"Baseline Macro-F1: {f1_macro.mean():.2f} (± {f1_macro.std():.2f})")


if __name__ == "__main__":
    main()

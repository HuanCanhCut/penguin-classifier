"""Thí nghiệm 1: KNN không scale và KNN có StandardScaler.

Cách chạy (từ thư mục gốc của project, sau khi chạy prepare_data.py):
    python ml/src/experiment_scaling.py

Đánh giá bằng cross-validation 5 fold trên tập train, cùng cách chia fold với baseline.py.
Kết quả được lưu vào reports/experiment_scaling.csv.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
RESULT_CSV = ML_DIR / "reports" / "experiment_scaling.csv"

TARGET = "species"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
RANDOM_STATE = 42


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    X = train[FEATURES]
    y = train[TARGET]

    # Hai version chỉ khác nhau ở bước scale, KNN giữ tham số mặc định (k=5)
    models = {
        "KNN raw": KNeighborsClassifier(),
        "KNN + scaler": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        # Scaler nằm trong Pipeline nên được fit lại trên phần train của từng fold, không học từ fold đánh giá
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Accuracy std": scores["test_accuracy"].std(),
            "Macro-F1": scores["test_f1_macro"].mean(),
            "Macro-F1 std": scores["test_f1_macro"].std(),
        })

    result = pd.DataFrame(rows)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(RESULT_CSV, index=False)

    print(f"Train: {len(train)} row, feature: {', '.join(FEATURES)}")
    print("Đánh giá: cross-validation 5 fold (stratified) trên tập train\n")
    print(f"{'Model':<14}{'Accuracy':>16}{'Macro-F1':>16}")
    for row in rows:
        print(
            f"{row['Model']:<14}"
            f"{row['Accuracy']:>9.2f} ± {row['Accuracy std']:.2f}"
            f"{row['Macro-F1']:>9.2f} ± {row['Macro-F1 std']:.2f}"
        )
    print(f"\nĐã lưu: {RESULT_CSV}")


if __name__ == "__main__":
    main()

"""Thí nghiệm 3: chỉ 4 số đo và 4 số đo + sex + island.

Cách chạy (từ thư mục gốc của project, sau khi chạy tune.py):
    python ml/src/experiment_features.py

Cả hai version dùng cùng config KNN tốt nhất trong reports/best_params.json và cùng cách chia fold
với các thí nghiệm trước. Kết quả được lưu vào reports/experiment_features.csv.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
BEST_PARAMS_JSON = ML_DIR / "reports" / "best_params.json"
RESULT_CSV = ML_DIR / "reports" / "experiment_features.csv"

TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["sex", "island"]
RANDOM_STATE = 42


def build_model(numeric: list[str], categorical: list[str], knn_params: dict) -> Pipeline:
    transformers = [("num", StandardScaler(), numeric)]
    if categorical:
        # handle_unknown="ignore": gặp giá trị lạ (ví dụ đảo mới) thì mã hoá thành toàn 0 thay vì báo lỗi
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("knn", KNeighborsClassifier(**knn_params)),
    ])


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    y = train[TARGET]

    best = json.loads(BEST_PARAMS_JSON.read_text())
    knn_params = {key: best[key] for key in ("n_neighbors", "weights", "metric")}

    feature_sets = {
        "Numeric only": (NUMERIC_FEATURES, []),
        "Numeric + sex + island": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, (numeric, categorical) in feature_sets.items():
        model = build_model(numeric, categorical, knn_params)
        # Scaler và encoder nằm trong Pipeline nên chỉ học từ phần train của từng fold
        scores = cross_validate(model, train[numeric + categorical], y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append({
            "Feature set": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Accuracy std": scores["test_accuracy"].std(),
            "Macro-F1": scores["test_f1_macro"].mean(),
            "Macro-F1 std": scores["test_f1_macro"].std(),
        })

    result = pd.DataFrame(rows)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(RESULT_CSV, index=False)

    print(f"Train: {len(train)} row, KNN: {knn_params}")
    print("Đánh giá: cross-validation 5 fold (stratified) trên tập train\n")
    print(f"{'Feature set':<26}{'Accuracy':>18}{'Macro-F1':>18}")
    for row in rows:
        print(
            f"{row['Feature set']:<26}"
            f"{row['Accuracy']:>9.4f} ± {row['Accuracy std']:.4f}"
            f"{row['Macro-F1']:>9.4f} ± {row['Macro-F1 std']:.4f}"
        )
    print(f"\nĐã lưu: {RESULT_CSV}")


if __name__ == "__main__":
    main()

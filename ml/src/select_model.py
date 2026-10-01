"""Chốt model candidate: chạy GridSearchCV cho từng bộ feature rồi chọn config tốt nhất.

Cách chạy (từ thư mục gốc của project, sau khi chạy prepare_data.py):
    python ml/src/select_model.py

Chọn theo Macro-F1 trung bình CV 5 fold trên tập train (cùng cách chia fold với các thí nghiệm trước).
Kết quả được lưu vào:
- reports/model_selection.csv: config tốt nhất của từng bộ feature
- config.json: config của model candidate, dùng để huấn luyện và đánh giá trên test
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from experiment_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_model
from tune import PARAM_GRID

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
RESULT_CSV = ML_DIR / "reports" / "model_selection.csv"
CONFIG_JSON = ML_DIR / "config.json"

TARGET = "species"
RANDOM_STATE = 42

# Xếp từ đơn giản đến phức tạp: khi hai bộ feature bằng điểm thì giữ bộ đơn giản hơn
FEATURE_SETS = {
    "numeric": (NUMERIC_FEATURES, []),
    "numeric+sex+island": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
}


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    y = train[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, (numeric, categorical) in FEATURE_SETS.items():
        search = GridSearchCV(
            build_model(numeric, categorical, {}),
            PARAM_GRID,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            refit="f1_macro",
            cv=cv,
        )
        search.fit(train[numeric + categorical], y)

        i = search.best_index_
        rows.append({
            "features": name,
            "n_neighbors": int(search.best_params_["knn__n_neighbors"]),
            "metric": search.best_params_["knn__metric"],
            "weights": search.best_params_["knn__weights"],
            "mean_cv_accuracy": search.cv_results_["mean_test_accuracy"][i],
            "std_cv_accuracy": search.cv_results_["std_test_accuracy"][i],
            "mean_cv_f1_macro": search.cv_results_["mean_test_f1_macro"][i],
            "std_cv_f1_macro": search.cv_results_["std_test_f1_macro"][i],
        })

    result = pd.DataFrame(rows)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.round(4).to_csv(RESULT_CSV, index=False)

    # Macro-F1 cao nhất thắng; bằng nhau thì std thấp hơn; vẫn bằng thì bộ feature đơn giản hơn (đứng trước)
    ranked = result.assign(order=range(len(result))).sort_values(
        ["mean_cv_f1_macro", "std_cv_f1_macro", "order"], ascending=[False, True, True]
    )
    best = ranked.iloc[0]
    numeric, categorical = FEATURE_SETS[best["features"]]

    config = {
        "n_neighbors": int(best["n_neighbors"]),
        "metric": best["metric"],
        "weights": best["weights"],
        "features": best["features"],
        "numeric_features": numeric,
        "categorical_features": categorical,
        "cv_accuracy": round(float(best["mean_cv_accuracy"]), 4),
        "cv_f1_macro": round(float(best["mean_cv_f1_macro"]), 4),
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n")

    print(f"Train: {len(train)} row, mỗi bộ feature thử {len(search.cv_results_['params'])} config\n")
    print("Config tốt nhất của từng bộ feature:")
    print(result.round(4).to_string(index=False))
    print("\nModel candidate:")
    print(json.dumps(config, indent=2))
    print(f"\nĐã lưu: {RESULT_CSV}")
    print(f"Đã lưu: {CONFIG_JSON}")


if __name__ == "__main__":
    main()

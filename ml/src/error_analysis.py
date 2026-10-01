"""Thí nghiệm 4: error analysis, xem K hàng xóm gần nhất của từng con bị đoán sai trên test.

Cách chạy (từ thư mục gốc của project, sau khi chạy evaluate.py):
    python ml/src/error_analysis.py

Dùng đúng model candidate trong config.json, huấn luyện trên toàn bộ train như evaluate.py.
Chỉ phân tích, không chỉnh model dựa trên kết quả này.
Kết quả được lưu vào reports/error_analysis.csv: mỗi con sai một dòng "sample", kèm K dòng "neighbor".
"""

import json
from pathlib import Path

import pandas as pd

from experiment_features import build_model

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
TEST_CSV = ML_DIR / "data" / "processed" / "test.csv"
CONFIG_JSON = ML_DIR / "config.json"
RESULT_CSV = ML_DIR / "reports" / "error_analysis.csv"

TARGET = "species"


def main() -> None:
    config = json.loads(CONFIG_JSON.read_text())
    features = config["numeric_features"] + config["categorical_features"]
    knn_params = {key: config[key] for key in ("n_neighbors", "weights", "metric")}

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    model = build_model(config["numeric_features"], config["categorical_features"], knn_params)
    model.fit(train[features], train[TARGET])
    predictions = model.predict(test[features])

    wrong = test[predictions != test[TARGET]].assign(predicted=predictions[predictions != test[TARGET]])
    print(f"Model: {knn_params}, features: {config['features']}")
    print(f"Số con đoán sai trên test: {len(wrong)}/{len(test)}")

    # Tìm hàng xóm trong không gian đã scale + one-hot, đúng không gian KNN dùng để dự đoán
    preprocess, knn = model.named_steps["preprocess"], model.named_steps["knn"]
    distances, indices = knn.kneighbors(preprocess.transform(wrong[features]))

    rows = []
    for case, ((_, sample), dist_row, idx_row) in enumerate(zip(wrong.iterrows(), distances, indices), start=1):
        rows.append({
            "case": case, "role": "sample", "rank": 0, "train_index": None, "distance": None,
            "species": sample[TARGET], "predicted": sample["predicted"], **sample[features].to_dict(),
        })
        for rank, (distance, index) in enumerate(zip(dist_row, idx_row), start=1):
            neighbor = train.iloc[index]
            rows.append({
                "case": case, "role": "neighbor", "rank": rank, "train_index": int(index),
                "distance": round(float(distance), 4), "species": neighbor[TARGET], "predicted": None,
                **neighbor[features].to_dict(),
            })

        neighbors = pd.DataFrame(rows[-len(idx_row):])
        # weights="distance": mỗi hàng xóm bỏ phiếu với trọng số 1/distance
        votes = (1 / neighbors["distance"]).groupby(neighbors["species"]).sum()
        print(f"\nCase {case}: Actual {sample[TARGET]} → Predicted {sample['predicted']}")
        print(neighbors.drop(columns=["case", "role", "predicted"]).to_string(index=False))
        print("Số hàng xóm theo loài:", neighbors["species"].value_counts().to_dict())
        print("Phiếu có trọng số 1/distance:", (votes / votes.sum()).round(3).to_dict())

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Int64 để train_index ghi ra 85 thay vì 85.0 (cột có ô trống ở dòng sample)
    pd.DataFrame(rows).astype({"train_index": "Int64"}).to_csv(RESULT_CSV, index=False)
    print(f"\nĐã lưu: {RESULT_CSV}")


if __name__ == "__main__":
    main()

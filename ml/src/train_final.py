"""Huấn luyện model cuối theo config.json và lưu pipeline cho backend.

Cách chạy (từ thư mục gốc của project, sau khi chạy select_model.py):
    python ml/src/train_final.py

Model được huấn luyện trên đúng tập train (266 row), giống hệt model đã chấm trong evaluate.py,
nên metrics trong reports/metrics.json mô tả đúng file được lưu.
Kết quả:
- models/penguin_pipeline.joblib: pipeline đầy đủ (scaler + one-hot + KNN), nhận DataFrame thô
- models/penguin_pipeline.json: thông tin đi kèm để backend biết cần gửi những cột nào
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from experiment_features import build_model

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
CONFIG_JSON = ML_DIR / "config.json"
METRICS_JSON = ML_DIR / "reports" / "metrics.json"
MODEL_PATH = ML_DIR / "models" / "penguin_pipeline.joblib"
MODEL_INFO_PATH = ML_DIR / "models" / "penguin_pipeline.json"

TARGET = "species"


def main() -> None:
    config = json.loads(CONFIG_JSON.read_text())
    features = config["numeric_features"] + config["categorical_features"]
    knn_params = {key: config[key] for key in ("n_neighbors", "weights", "metric")}

    train = pd.read_csv(TRAIN_CSV)
    pipeline = build_model(config["numeric_features"], config["categorical_features"], knn_params)
    pipeline.fit(train[features], train[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metrics = json.loads(METRICS_JSON.read_text())
    # Chỉ ghi các giá trị category đã gặp lúc train để backend kiểm tra input
    categories = {
        column: sorted(train[column].unique().tolist())
        for column in config["categorical_features"]
    }
    MODEL_INFO_PATH.write_text(json.dumps({
        "model_file": MODEL_PATH.name,
        "input_features": features,
        "numeric_features": config["numeric_features"],
        "categorical_features": categories,
        "classes": pipeline.classes_.tolist(),
        "knn": knn_params,
        "n_train": len(train),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        # Load pipeline bằng phiên bản scikit-learn khác có thể lỗi hoặc cho kết quả sai
        "sklearn_version": sklearn.__version__,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} row, model: {knn_params}, features: {features}")
    print(f"Đã lưu: {MODEL_PATH}")
    print(f"Đã lưu: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    main()

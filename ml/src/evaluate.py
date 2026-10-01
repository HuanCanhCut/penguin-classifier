"""Đánh giá cuối: huấn luyện model candidate trên toàn bộ train rồi chấm MỘT lần trên test.

Cách chạy (từ thư mục gốc của project, sau khi chạy select_model.py):
    python ml/src/evaluate.py

Model lấy từ config.json, không chỉnh gì thêm. Sau bước này không tune model dựa trên test nữa.
Kết quả được lưu vào:
- reports/metrics.json: accuracy, macro-F1, confusion matrix, classification report
- reports/confusion_matrix.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from experiment_features import build_model

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
TEST_CSV = ML_DIR / "data" / "processed" / "test.csv"
CONFIG_JSON = ML_DIR / "config.json"
METRICS_JSON = ML_DIR / "reports" / "metrics.json"
CONFUSION_PNG = ML_DIR / "reports" / "confusion_matrix.png"

TARGET = "species"
LABELS = ["Adelie", "Chinstrap", "Gentoo"]


def plot_confusion_matrix(matrix, accuracy: float, macro_f1: float) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Loài dự đoán")
    ax.set_ylabel("Loài thực tế")
    ax.set_title(f"Confusion matrix trên test\nAccuracy {accuracy:.4f} · Macro-F1 {macro_f1:.4f}")
    fig.tight_layout()

    CONFUSION_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_JSON.read_text())
    features = config["numeric_features"] + config["categorical_features"]
    knn_params = {key: config[key] for key in ("n_neighbors", "weights", "metric")}

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # Scaler và encoder chỉ học từ train; test chỉ đi qua transform rồi predict
    model = build_model(config["numeric_features"], config["categorical_features"], knn_params)
    model.fit(train[features], train[TARGET])
    predictions = model.predict(test[features])

    y_test = test[TARGET]
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    report = classification_report(y_test, predictions, labels=LABELS, output_dict=True)

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps({
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "n_train": len(train),
        "n_test": len(test),
        "model": {**knn_params, "features": config["features"]},
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": {
            label: {metric: round(value, 4) for metric, value in scores.items()}
            if isinstance(scores, dict) else round(scores, 4)
            for label, scores in report.items()
        },
    }, indent=2) + "\n")

    plot_confusion_matrix(matrix, accuracy, macro_f1)

    print(f"Model: {knn_params}, features: {config['features']}")
    print(f"Train: {len(train)} row, test: {len(test)} row\n")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro-F1: {macro_f1:.4f}")
    print(f"CV Macro-F1 (lúc chọn model): {config['cv_f1_macro']:.4f}\n")
    print("Confusion matrix (hàng: thực tế, cột: dự đoán):")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=LABELS, digits=4))

    wrong = test[predictions != y_test].assign(predicted=predictions[predictions != y_test])
    if not wrong.empty:
        print("Các con đoán sai:")
        print(wrong.to_string(index=False))

    print(f"\nĐã lưu: {METRICS_JSON}")
    print(f"Đã lưu: {CONFUSION_PNG}")


if __name__ == "__main__":
    main()

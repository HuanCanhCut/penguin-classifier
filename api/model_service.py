"""Load Pipeline đã train offline (ml/src/train_final.py) và dự đoán. API không bao giờ gọi fit()."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "ml"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ML_DIR / "models" / "penguin_pipeline.joblib"))
MODEL_INFO_PATH = MODEL_PATH.with_suffix(".json")
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
CLEAN_CSV = ML_DIR / "data" / "processed" / "penguins_clean.csv"
METRICS_JSON = ML_DIR / "reports" / "metrics.json"

N_NEIGHBORS_SHOWN = 5

# Khoảng thường gặp, lấy tròn từ min/max của dữ liệu sạch (333 con). Ngoài khoảng này vẫn dự đoán
# nhưng kèm cảnh báo: KNN chỉ đáng tin khi điểm mới nằm gần vùng dữ liệu đã thấy.
TYPICAL_RANGES = {
    "bill_length_mm": (32.0, 60.0),
    "bill_depth_mm": (13.0, 21.5),
    "flipper_length_mm": (170.0, 232.0),
    "body_mass_g": (2700.0, 6300.0),
}


@dataclass
class ModelService:
    pipeline: Pipeline
    info: dict
    train: pd.DataFrame | None

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH) -> "ModelService":
        pipeline = joblib.load(model_path)
        info = json.loads(model_path.with_suffix(".json").read_text())

        # train.csv chỉ dùng để hiển thị số đo của hàng xóm; chỉ tin khi khớp đúng số row model đã fit
        train = pd.read_csv(TRAIN_CSV) if TRAIN_CSV.exists() else None
        if train is not None and len(train) != pipeline.named_steps["knn"].n_samples_fit_:
            train = None
        return cls(pipeline=pipeline, info=info, train=train)

    @property
    def classes(self) -> list[str]:
        return self.pipeline.classes_.tolist()

    def predict(self, features: dict) -> dict:
        X = pd.DataFrame([features], columns=self.info["input_features"])
        knn = self.pipeline.named_steps["knn"]

        proba = self.pipeline.predict_proba(X)[0]
        prediction = self.classes[int(np.argmax(proba))]

        X_transformed = self.pipeline.named_steps["preprocess"].transform(X)
        distances, indices = knn.kneighbors(X_transformed, n_neighbors=N_NEIGHBORS_SHOWN)
        # knn._y giữ nhãn (đã mã hoá) của từng điểm train theo đúng thứ tự lúc fit
        neighbor_species = knn.classes_[knn._y[indices[0]]]

        neighbors = []
        for rank, (distance, index, species) in enumerate(
            zip(distances[0], indices[0], neighbor_species), start=1
        ):
            neighbors.append({
                "rank": rank,
                "species": str(species),
                "distance": round(float(distance), 4),
                "features": self._train_features(int(index)),
            })

        votes = {name: 0 for name in self.classes}
        for species in neighbor_species:
            votes[str(species)] += 1

        return {
            "prediction": prediction,
            "confidence": {name: round(float(p), 4) for name, p in zip(self.classes, proba)},
            "neighbor_votes": votes,
            "neighbors": neighbors,
            "warnings": typical_range_warnings(features),
        }

    def _train_features(self, index: int) -> dict | None:
        if self.train is None:
            return None
        row = self.train.iloc[index]
        return {column: _to_json(row[column]) for column in self.info["input_features"]}


def typical_range_warnings(features: dict) -> list[dict]:
    warnings = []
    for field, (low, high) in TYPICAL_RANGES.items():
        value = features[field]
        if not low <= value <= high:
            warnings.append({
                "field": field,
                "message": f"{value:g} nằm ngoài khoảng thường gặp [{low:g}, {high:g}], "
                           "kết quả có thể kém tin cậy",
            })
    return warnings


def dataset_stats() -> dict:
    """Thống kê cho dashboard, đọc từ dữ liệu đã làm sạch (không liên quan tới việc train)."""
    data = pd.read_csv(CLEAN_CSV)
    numeric = list(TYPICAL_RANGES)

    species_counts = data["species"].value_counts()
    by_island = pd.crosstab(data["island"], data["species"])
    means = data.groupby("species")[numeric].mean().round(1)

    return {
        "n_samples": len(data),
        "species_distribution": {k: int(v) for k, v in species_counts.items()},
        "island_distribution": {
            island: {species: int(n) for species, n in row.items()}
            for island, row in by_island.iterrows()
        },
        "feature_means": {
            species: {k: float(v) for k, v in row.items()} for species, row in means.iterrows()
        },
        "feature_ranges": {
            column: {"min": float(data[column].min()), "max": float(data[column].max())}
            for column in numeric
        },
        "points": data[["species", "bill_length_mm", "bill_depth_mm"]].to_dict(orient="records"),
    }


def model_metrics() -> dict | None:
    return json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else None


def _to_json(value):
    return value.item() if hasattr(value, "item") else value

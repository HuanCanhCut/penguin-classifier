"""Chọn config KNN bằng GridSearchCV: số láng giềng K, cách đánh trọng số, độ đo khoảng cách.

Cách chạy (từ thư mục gốc của project, sau khi chạy prepare_data.py):
    python ml/src/tune.py

Đánh giá bằng cross-validation 5 fold trên tập train, cùng cách chia fold với baseline.py.
Kết quả được lưu vào:
- reports/tuning_results.csv: điểm của tất cả config
- reports/best_params.json: config tốt nhất, dùng cho bước huấn luyện model cuối
- reports/figures/k_vs_cv_score.png: biểu đồ K và điểm CV
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ML_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = ML_DIR / "data" / "processed" / "train.csv"
RESULT_CSV = ML_DIR / "reports" / "tuning_results.csv"
BEST_PARAMS_JSON = ML_DIR / "reports" / "best_params.json"
FIGURE_PNG = ML_DIR / "reports" / "figures" / "k_vs_cv_score.png"

TARGET = "species"
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
RANDOM_STATE = 42

PARAM_GRID = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}


WEIGHT_STYLES = {
    "uniform": {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    "distance": {"color": "#eb6834", "marker": "s", "linestyle": "--"},
}


def plot_k_vs_score(results: pd.DataFrame, best: dict) -> None:
    """Vẽ Macro-F1 CV theo K, mỗi metric một ô, mỗi weights một đường, dải mờ là ± std."""
    metrics = ["euclidean", "manhattan"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4.2), sharey=True)

    for ax, metric in zip(axes, metrics):
        for weights, style in WEIGHT_STYLES.items():
            part = results[(results["metric"] == metric) & (results["weights"] == weights)].sort_values("k")
            mean, std = part["mean_cv_f1_macro"], part["std_cv_f1_macro"]
            ax.plot(part["k"], mean, label=weights, linewidth=2, markersize=6, **style)
            # Macro-F1 không vượt quá 1 nên cắt dải ở 1
            ax.fill_between(part["k"], mean - std, (mean + std).clip(upper=1), color=style["color"], alpha=0.12, linewidth=0)

        if metric == best["metric"]:
            best_score = results[
                (results["k"] == best["k"]) & (results["metric"] == metric) & (results["weights"] == best["weights"])
            ]["mean_cv_f1_macro"].iloc[0]
            ax.annotate(
                f"Best: K={best['k']}, {best['weights']}\nMacro-F1 = {best_score:.4f}",
                xy=(best["k"], best_score), xytext=(best["k"] + 2.5, best_score + 0.012),
                arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=9,
            )

        ax.set_title(f"metric = {metric}")
        ax.set_xlabel("K (n_neighbors)")
        ax.set_xticks(PARAM_GRID["knn__n_neighbors"])
        ax.grid(color="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Macro-F1 (CV 5 fold)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="weights", frameon=False, loc="upper right", ncols=2)
    fig.suptitle("Điểm CV theo K (dải mờ: ± độ lệch chuẩn giữa các fold)")
    fig.tight_layout()

    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(TRAIN_CSV)
    X = train[FEATURES]
    y = train[TARGET]

    # Thí nghiệm 1 cho thấy KNN cần scale, nên mọi config đều có StandardScaler trong Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        # Chọn config theo Macro-F1, chỉ số chính của project
        refit="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    search.fit(X, y)

    cv = search.cv_results_
    results = pd.DataFrame({
        "k": cv["param_knn__n_neighbors"].astype(int),
        "metric": cv["param_knn__metric"],
        "weights": cv["param_knn__weights"],
        "mean_cv_accuracy": cv["mean_test_accuracy"],
        "std_cv_accuracy": cv["std_test_accuracy"],
        "mean_cv_f1_macro": cv["mean_test_f1_macro"],
        "std_cv_f1_macro": cv["std_test_f1_macro"],
        "rank_f1_macro": cv["rank_test_f1_macro"],
    })
    results["weights"] = pd.Categorical(results["weights"], PARAM_GRID["knn__weights"])
    results = results.sort_values(["k", "metric", "weights"])
    results["weights"] = results["weights"].astype(str)

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(RESULT_CSV, index=False)

    best = {
        "k": int(search.best_params_["knn__n_neighbors"]),
        "weights": search.best_params_["knn__weights"],
        "metric": search.best_params_["knn__metric"],
    }
    best_row = results.loc[results["rank_f1_macro"] == 1].iloc[0]
    BEST_PARAMS_JSON.write_text(json.dumps({
        "n_neighbors": best["k"],
        "weights": best["weights"],
        "metric": best["metric"],
        "cv_accuracy": round(float(best_row["mean_cv_accuracy"]), 4),
        "cv_f1_macro": round(float(best_row["mean_cv_f1_macro"]), 4),
    }, indent=2) + "\n")

    plot_k_vs_score(results, best)

    print(f"Train: {len(train)} row, đã thử {len(results)} config, chọn theo Macro-F1 (CV 5 fold)\n")
    print("Best params:")
    print(f"  K = {best['k']}")
    print(f"  weights = {best['weights']}")
    print(f"  metric = {best['metric']}")
    print(f"  Accuracy: {best_row['mean_cv_accuracy']:.4f} ± {best_row['std_cv_accuracy']:.4f}")
    print(f"  Macro-F1: {best_row['mean_cv_f1_macro']:.4f} ± {best_row['std_cv_f1_macro']:.4f}")

    print("\nTop 10 config:")
    print(results.sort_values(["rank_f1_macro", "k"]).head(10).round(4).to_string(index=False))
    print(f"\nĐã lưu: {RESULT_CSV}")
    print(f"Đã lưu: {BEST_PARAMS_JSON}")
    print(f"Đã lưu: {FIGURE_PNG}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT


FEATURE_COLUMNS = [
    "age_hours",
    "current_temp",
    "mean_temp",
    "min_temp",
    "max_temp",
    "temp_std",
    "minutes_above_threshold",
    "excursion_count",
    "longest_excursion",
    "degree_hours",
    "horizon_hours",
    "warming_rate",
]


def train_xgb(
    features_csv: str | Path,
    model_path: str | Path = PROJECT_ROOT / "models" / "xgboost_thermal.json",
    metrics_path: str | Path = PROJECT_ROOT / "outputs" / "metrics" / "xgb_metrics.json",
    plot_path: str | Path = PROJECT_ROOT / "outputs" / "plots" / "confusion_matrix_xgb.png",
) -> dict:
    df = pd.read_csv(features_csv)
    missing = set(FEATURE_COLUMNS + ["downgrade"]) - set(df.columns)
    if missing:
        raise ValueError(f"thermal_features.csv missing columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS]
    y = df["downgrade"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_rows": int(len(y_test)),
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    plot_path = Path(plot_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(model_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm, display_labels=["no_downgrade", "downgrade"]).plot(values_format="d")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=PROJECT_ROOT / "data" / "thermal" / "thermal_features.csv")
    args = parser.parse_args()
    print(json.dumps(train_xgb(args.features), indent=2))


if __name__ == "__main__":
    main()

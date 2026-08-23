import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.config import PROJECT_ROOT
from src.tabular.dataset import RAW_ROOT, load_competition_table, make_feature_matrix


MODEL_DIR = PROJECT_ROOT / "models" / "tabular"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"


def _fit_classifier(X_train, y_train, num_class: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _save_confusion_matrix(y_true, y_pred, labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(values_format="d", xticks_rotation=35)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_competition_models(raw_root: str | Path = RAW_ROOT) -> dict:
    df = load_competition_table(raw_root)
    X = make_feature_matrix(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["recommended_action"],
    )

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    action_encoder = LabelEncoder()
    action_y = action_encoder.fit_transform(df["recommended_action"])
    action_model = _fit_classifier(X_train, action_y[train_idx], len(action_encoder.classes_))
    action_pred = action_model.predict(X_test)

    condition_encoder = LabelEncoder()
    condition_y = condition_encoder.fit_transform(df["condition_status"])
    condition_model = _fit_classifier(X_train, condition_y[train_idx], len(condition_encoder.classes_))
    condition_pred = condition_model.predict(X_test)

    risk_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    risk_model.fit(X_train, df.loc[train_idx, "risk_score_0_100"])
    risk_pred = risk_model.predict(X_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    action_model.save_model(MODEL_DIR / "action_xgb.json")
    condition_model.save_model(MODEL_DIR / "condition_xgb.json")
    risk_model.save_model(MODEL_DIR / "risk_xgb.json")

    metadata = {
        "feature_columns": list(X.columns),
        "action_classes": action_encoder.classes_.tolist(),
        "condition_classes": condition_encoder.classes_.tolist(),
        "source_dataset": str(Path(raw_root)),
        "notes": [
            "Dataset labels are synthetic/weak/scenario labels from the AIC package.",
            "Predictions are decision-support outputs, not food-safety or certification claims.",
        ],
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    action_labels = action_encoder.inverse_transform(action_pred)
    condition_labels = condition_encoder.inverse_transform(condition_pred)
    true_action = df.loc[test_idx, "recommended_action"].to_numpy()
    true_condition = df.loc[test_idx, "condition_status"].to_numpy()

    _save_confusion_matrix(true_action, action_labels, metadata["action_classes"], PLOT_DIR / "confusion_matrix_action.png")
    _save_confusion_matrix(true_condition, condition_labels, metadata["condition_classes"], PLOT_DIR / "confusion_matrix_condition.png")

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "action_accuracy": float(accuracy_score(true_action, action_labels)),
        "condition_accuracy": float(accuracy_score(true_condition, condition_labels)),
        "risk_mae": float(mean_absolute_error(df.loc[test_idx, "risk_score_0_100"], risk_pred)),
        "risk_r2": float(r2_score(df.loc[test_idx, "risk_score_0_100"], risk_pred)),
        "action_report": classification_report(true_action, action_labels, output_dict=True, zero_division=0),
        "condition_report": classification_report(true_condition, condition_labels, output_dict=True, zero_division=0),
    }
    with open(OUTPUT_DIR / "competition_model_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=RAW_ROOT)
    args = parser.parse_args()
    metrics = train_competition_models(args.raw_root)
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("_report")}, indent=2))


if __name__ == "__main__":
    main()

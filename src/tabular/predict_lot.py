import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import PROJECT_ROOT
from src.tabular.dataset import RAW_ROOT, load_available_table, make_feature_matrix


MODEL_DIR = PROJECT_ROOT / "models" / "tabular"


def _load_metadata(model_dir: str | Path = MODEL_DIR) -> dict:
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Tabular models are missing. Run: python -m src.tabular.train_models")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _top_probabilities(classes: list[str], probabilities, limit: int = 3) -> list[dict]:
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:limit]
    return [{"label": label, "probability": float(prob)} for label, prob in ranked]


def predict_feature_row(row: pd.DataFrame, model_dir: str | Path = MODEL_DIR) -> dict:
    metadata = _load_metadata(model_dir)
    X = make_feature_matrix(row, expected_columns=metadata["feature_columns"])

    matrix = xgb.DMatrix(X, feature_names=list(X.columns))

    action_model = xgb.Booster()
    action_model.load_model(Path(model_dir) / "action_xgb.json")
    condition_model = xgb.Booster()
    condition_model.load_model(Path(model_dir) / "condition_xgb.json")
    risk_model = xgb.Booster()
    risk_model.load_model(Path(model_dir) / "risk_xgb.json")

    action_prob = np.asarray(action_model.predict(matrix))[0]
    condition_prob = np.asarray(condition_model.predict(matrix))[0]
    risk_score = float(np.asarray(risk_model.predict(matrix))[0])

    action_top = _top_probabilities(metadata["action_classes"], action_prob)
    condition_top = _top_probabilities(metadata["condition_classes"], condition_prob)

    actual = row.iloc[0]
    return {
        "lot_id": actual.get("lot_id", "MANUAL-LOT"),
        "predicted_risk_score_0_100": risk_score,
        "predicted_action": action_top[0]["label"],
        "action_probabilities": action_top,
        "predicted_condition": condition_top[0]["label"],
        "condition_probabilities": condition_top,
        "dataset_reference": {
            "weak_label_action": actual.get("recommended_action"),
            "weak_label_risk_score_0_100": float(actual.get("risk_score_0_100")),
            "proxy_visual_score_0_16": float(actual.get("total_proxy_0_16")),
            "remaining_quality_window_h": float(actual.get("remaining_quality_window_h")),
            "max_temp_c": float(actual.get("max_temp_c")),
            "time_above_10c_h": float(actual.get("time_above_10c_h")),
            "handling_scenario": actual.get("handling_scenario"),
        },
        "claim_note": "Decision-support prediction only; not food-safety, SNI, histamine, or export certification.",
    }


def predict_lot(lot_id: str, raw_root: str | Path = RAW_ROOT, model_dir: str | Path = MODEL_DIR) -> dict:
    df = load_available_table(raw_root)
    row = df[df["lot_id"] == lot_id]
    if row.empty:
        raise ValueError(f"Unknown lot_id: {lot_id}")
    return predict_feature_row(row, model_dir)


def print_prediction(result: dict) -> None:
    ref = result["dataset_reference"]
    print("\n========================================")
    print("LEMURU LOGISTICS AI")
    print("========================================\n")
    print(f"LOT                 : {result['lot_id']}")
    print(f"Condition           : {result['predicted_condition']}")
    print(f"Risk score          : {result['predicted_risk_score_0_100']:.1f} / 100")
    print(f"Recommended action  : {result['predicted_action']}\n")
    print("ACTION PROBABILITY")
    for item in result["action_probabilities"]:
        print(f"- {item['label']:<20} {item['probability'] * 100:5.1f}%")
    print("\nCONDITION PROBABILITY")
    for item in result["condition_probabilities"]:
        print(f"- {item['label']:<20} {item['probability'] * 100:5.1f}%")
    print("\nKEY SIGNALS")
    print(f"- Remaining quality window : {ref['remaining_quality_window_h']:.1f} h")
    print(f"- Max temperature          : {ref['max_temp_c']:.1f} C")
    print(f"- Time above 10 C          : {ref['time_above_10c_h']:.1f} h")
    print(f"- Visual proxy score       : {ref['proxy_visual_score_0_16']:.0f} / 16")
    print(f"- Handling scenario        : {ref['handling_scenario']}")
    print("\nNOTE")
    print(result["claim_note"])
    print("========================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    result = predict_lot(args.lot_id, args.raw_root)
    print_prediction(result)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()

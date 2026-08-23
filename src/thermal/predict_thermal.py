import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.config import PROJECT_ROOT, load_config
from src.thermal.feature_engineering import extract_thermal_features, load_temperature_csv
from src.thermal.train_xgb import FEATURE_COLUMNS


def risk_band(probability: float, low_threshold: float, high_threshold: float) -> str:
    if probability < low_threshold:
        return "LOW"
    if probability < high_threshold:
        return "WATCH"
    return "HIGH"


def predict_thermal(
    temperature_csv: str | Path,
    lot_id: str,
    age_hours: float,
    horizon_hours: float,
    model_path: str | Path = PROJECT_ROOT / "models" / "xgboost_thermal.json",
    config_path: str | Path | None = None,
) -> dict:
    config = load_config(config_path or PROJECT_ROOT / "config" / "config.yaml")
    threshold = float(config["thermal"]["temperature_threshold"])
    low = float(config["thermal"]["low_risk_threshold"])
    high = float(config["thermal"]["high_risk_threshold"])

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError("Thermal model is missing. Run: python -m src.thermal.train_xgb")

    df = load_temperature_csv(temperature_csv)
    features = extract_thermal_features(df, lot_id, age_hours, horizon_hours, threshold)
    feature_frame = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])

    model = XGBClassifier()
    model.load_model(model_path)
    probability = float(model.predict_proba(feature_frame)[0, 1])

    return {
        "downgrade_probability": probability,
        "risk_band": risk_band(probability, low, high),
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--lot-id", required=True)
    parser.add_argument("--age-hours", type=float, required=True)
    parser.add_argument("--horizon-hours", type=float, required=True)
    args = parser.parse_args()
    result = predict_thermal(args.temperature, args.lot_id, args.age_hours, args.horizon_hours)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

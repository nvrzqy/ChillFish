import pandas as pd

from src.thermal.feature_engineering import extract_thermal_features


def test_extract_thermal_features_counts_single_excursion():
    df = pd.DataFrame(
        {
            "lot_id": ["L001"] * 5,
            "timestamp": pd.date_range("2026-08-20 08:00", periods=5, freq="h"),
            "temperature": [-18.0, -14.0, -13.0, -18.0, -17.0],
        }
    )

    features = extract_thermal_features(df, "L001", age_hours=5, horizon_hours=48, temperature_threshold=-15)

    assert features["excursion_count"] == 1
    assert features["minutes_above_threshold"] == 120.0
    assert features["degree_hours"] == 3.0

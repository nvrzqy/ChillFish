from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "aic_dataset_package"
CORE_ROOT = RAW_ROOT / "01_core_network"
APP_TABLE_PATH = PROJECT_ROOT / "data" / "app" / "competition_inference_table.csv"

NUMERIC_FEATURES = [
    "lot_mass_kg",
    "prehistory_equivalent_ice_h",
    "baseline_shelf_life_h",
    "min_temp_c",
    "mean_temp_c",
    "max_temp_c",
    "time_above_4c_h",
    "time_above_10c_h",
    "time_above_15c_h",
    "equivalent_ice_age_h",
    "remaining_quality_window_h",
    "logger_missing_intervals",
    "eye_demerit_0_3",
    "gill_demerit_0_3",
    "odor_demerit_0_3",
    "texture_demerit_0_3",
    "mucus_demerit_0_3",
    "belly_burst_flag",
    "total_proxy_0_16",
]

CATEGORICAL_FEATURES = [
    "product_form",
    "handling_scenario",
    "traceability_status",
    "origin_node_id",
    "target_market_node_id",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def condition_from_proxy(score: float) -> str:
    if score <= 5:
        return "NORMAL"
    if score <= 10:
        return "CHECK"
    return "POOR"


def load_competition_table(raw_root: str | Path = RAW_ROOT) -> pd.DataFrame:
    raw_root = Path(raw_root)
    core = raw_root / "01_core_network"
    if not core.exists():
        raise FileNotFoundError(f"Dataset folder not found: {core}")

    fish_lots = pd.read_csv(core / "fish_lots.csv")
    thermal = pd.read_csv(core / "thermal_features.csv")
    visual = pd.read_csv(core / "structured_visual_observations.csv")
    decisions = pd.read_csv(core / "decision_labels.csv")

    df = fish_lots.merge(thermal, on="lot_id", how="inner")
    df = df.merge(visual, on="lot_id", how="inner", suffixes=("", "_visual"))
    df = df.merge(decisions, on="lot_id", how="inner", suffixes=("", "_decision"))

    df["condition_status"] = df["total_proxy_0_16"].apply(condition_from_proxy)
    return df


def load_app_table(path: str | Path = APP_TABLE_PATH) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Packaged inference table not found: {table_path}. "
            "Run: python -m src.tabular.export_app_table"
        )
    return pd.read_csv(table_path)


def load_available_table(raw_root: str | Path = RAW_ROOT) -> pd.DataFrame:
    try:
        return load_competition_table(raw_root)
    except FileNotFoundError:
        return load_app_table()


def make_feature_matrix(df: pd.DataFrame, expected_columns: list[str] | None = None) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    features = df[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("UNKNOWN").astype(str)

    matrix = pd.get_dummies(features, columns=CATEGORICAL_FEATURES, dummy_na=False)
    if expected_columns is not None:
        matrix = matrix.reindex(columns=expected_columns, fill_value=0)
    return matrix.astype(float)

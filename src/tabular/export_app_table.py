from pathlib import Path

from src.config import PROJECT_ROOT
from src.tabular.dataset import APP_TABLE_PATH, load_competition_table


KEEP_COLUMNS = [
    "lot_id",
    "species_common",
    "scientific_name",
    "lot_mass_kg",
    "product_form",
    "handling_scenario",
    "prehistory_equivalent_ice_h",
    "baseline_shelf_life_h",
    "target_market_node_id",
    "traceability_status",
    "origin_node_id",
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
    "condition_status",
    "risk_score_0_100",
    "recommended_action",
    "candidate_destination_node_id",
    "rationale",
    "human_override_required",
]


def export_app_table(output_path: str | Path = APP_TABLE_PATH) -> Path:
    df = load_competition_table()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[KEEP_COLUMNS].to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = export_app_table()
    relative = output_path.relative_to(PROJECT_ROOT)
    print(f"Exported packaged inference table: {relative}")


if __name__ == "__main__":
    main()

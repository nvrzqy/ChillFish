import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.thermal.feature_engineering import extract_thermal_features


SEED = 42
PROFILE_TYPES = ["stable", "mild_excursion", "repeated_excursion", "severe_excursion", "gradual_warming"]


def _make_profile(profile_type: str, lot_id: str, start: pd.Timestamp, hours: int) -> pd.DataFrame:
    seed_bytes = f"{profile_type}:{lot_id}:{SEED}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
    rng = np.random.default_rng(stable_seed)
    timestamps = pd.date_range(start=start, periods=hours + 1, freq="h")
    base = -18.0 + rng.normal(0, 0.35, size=len(timestamps))

    if profile_type == "stable":
        temps = base
    elif profile_type == "mild_excursion":
        temps = base
        start_idx = rng.integers(4, max(5, hours - 4))
        temps[start_idx : start_idx + 3] += rng.uniform(3.0, 5.0)
    elif profile_type == "repeated_excursion":
        temps = base
        for _ in range(3):
            start_idx = rng.integers(2, max(3, hours - 3))
            temps[start_idx : start_idx + 2] += rng.uniform(3.5, 6.5)
    elif profile_type == "severe_excursion":
        temps = base
        start_idx = rng.integers(4, max(5, hours - 8))
        temps[start_idx : start_idx + 7] += rng.uniform(6.5, 10.0)
    elif profile_type == "gradual_warming":
        temps = base + np.linspace(0, rng.uniform(4.0, 8.0), len(timestamps))
    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")

    return pd.DataFrame({"lot_id": lot_id, "timestamp": timestamps, "temperature": temps.round(2)})


def _label_from_features(features: dict) -> int:
    burden = (
        0.035 * features["degree_hours"]
        + 0.006 * features["minutes_above_threshold"]
        + 0.12 * features["excursion_count"]
        + 0.01 * features["horizon_hours"]
        + 0.10 * max(0.0, features["warming_rate"])
    )
    return int(burden >= 1.0)


def generate_dataset(
    output_raw: str | Path,
    output_features: str | Path,
    lots_per_type: int = 80,
    temperature_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(SEED)
    np.random.seed(SEED)
    config = load_config()
    threshold = float(temperature_threshold or config["thermal"]["temperature_threshold"])

    raw_frames = []
    feature_rows = []
    start = pd.Timestamp("2026-08-20 08:00")
    index = 1

    for profile_type in PROFILE_TYPES:
        for _ in range(lots_per_type):
            lot_id = f"L{index:04d}"
            age_hours = int(np.random.randint(18, 73))
            horizon_hours = int(np.random.choice([24, 48, 72, 96]))
            raw = _make_profile(profile_type, lot_id, start, age_hours)
            raw_frames.append(raw)

            features = extract_thermal_features(raw, lot_id, age_hours, horizon_hours, threshold)
            features["profile_type"] = profile_type
            features["downgrade"] = _label_from_features(features)
            feature_rows.append(features)
            index += 1

    raw_df = pd.concat(raw_frames, ignore_index=True)
    features_df = pd.DataFrame(feature_rows)

    output_raw = Path(output_raw)
    output_features = Path(output_features)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_raw, index=False)
    features_df.to_csv(output_features, index=False)
    return raw_df, features_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-raw", default=PROJECT_ROOT / "data" / "thermal" / "temperature_raw.csv")
    parser.add_argument("--output-features", default=PROJECT_ROOT / "data" / "thermal" / "thermal_features.csv")
    parser.add_argument("--lots-per-type", type=int, default=80)
    args = parser.parse_args()

    _, features = generate_dataset(args.output_raw, args.output_features, args.lots_per_type)
    print(f"Generated {len(features)} thermal feature rows.")


if __name__ == "__main__":
    main()

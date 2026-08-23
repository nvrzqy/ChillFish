from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"lot_id", "timestamp", "temperature"}


def load_temperature_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Temperature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"temperature.csv missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("temperature.csv is empty")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="raise")
    if df["temperature"].isna().any():
        raise ValueError("temperature.csv contains NaN temperature values")

    return df.sort_values(["lot_id", "timestamp"]).reset_index(drop=True)


def _interval_minutes(timestamps: pd.Series) -> np.ndarray:
    if len(timestamps) <= 1:
        return np.array([60.0])

    deltas = timestamps.diff().dt.total_seconds().div(60).to_numpy().copy()
    fallback = float(np.nanmedian(deltas[1:])) if len(deltas) > 1 else 60.0
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 60.0
    deltas[0] = fallback
    deltas = np.where((deltas > 0) & np.isfinite(deltas), deltas, fallback)
    return deltas


def _excursion_stats(is_excursion: np.ndarray, interval_minutes: np.ndarray) -> tuple[int, float]:
    count = 0
    longest = 0.0
    current = 0.0
    in_event = False

    for flag, minutes in zip(is_excursion, interval_minutes):
        if flag:
            if not in_event:
                count += 1
                in_event = True
                current = 0.0
            current += float(minutes)
            longest = max(longest, current)
        else:
            in_event = False
            current = 0.0

    return count, longest


def extract_thermal_features(
    df: pd.DataFrame,
    lot_id: str,
    age_hours: float,
    horizon_hours: float,
    temperature_threshold: float,
) -> dict:
    lot_df = df[df["lot_id"] == lot_id].sort_values("timestamp").copy()
    if lot_df.empty:
        raise ValueError(f"No temperature rows found for lot_id={lot_id}")

    temps = lot_df["temperature"].to_numpy(dtype=float)
    intervals = _interval_minutes(lot_df["timestamp"])
    is_excursion = temps > float(temperature_threshold)
    excess = np.maximum(0.0, temps - float(temperature_threshold))

    excursion_count, longest_excursion = _excursion_stats(is_excursion, intervals)
    minutes_above = float(intervals[is_excursion].sum())
    degree_hours = float((excess * (intervals / 60.0)).sum())

    elapsed_hours = max(
        (lot_df["timestamp"].iloc[-1] - lot_df["timestamp"].iloc[0]).total_seconds() / 3600.0,
        1e-9,
    )
    warming_rate = float((temps[-1] - temps[0]) / elapsed_hours)

    return {
        "lot_id": lot_id,
        "age_hours": float(age_hours),
        "current_temp": float(temps[-1]),
        "mean_temp": float(np.mean(temps)),
        "min_temp": float(np.min(temps)),
        "max_temp": float(np.max(temps)),
        "temp_std": float(np.std(temps)),
        "minutes_above_threshold": minutes_above,
        "excursion_count": int(excursion_count),
        "longest_excursion": float(longest_excursion),
        "degree_hours": degree_hours,
        "horizon_hours": float(horizon_hours),
        "warming_rate": warming_rate,
    }

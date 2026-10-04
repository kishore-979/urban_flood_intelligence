"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features and keep the logic explainable."""
    df = df.copy()

    # Risk-oriented engineered features.
    df["rainfall_risk_flag"] = (df["rainfall_mm"] > 60).astype(int)
    df["low_drainage_flag"] = (df["drainage_index"] < 0.45).astype(int)
    df["steep_slope_flag"] = (df["slope_deg"] > 18).astype(int)
    df["high_history_flag"] = (df["local_flood_history"] > 1).astype(int)
    df["road_risk_flag"] = (df["road_quality"] < 0.5).astype(int)

    # A simple composite feature to mimic risk exposure.
    df["flood_exposure_index"] = (
        0.4 * df["rainfall_mm"] / 120
        + 0.2 * (1 - df["drainage_index"])
        + 0.2 * df["slope_deg"] / 30
        + 0.2 * df["proximity_to_channel"]
    )

    return df

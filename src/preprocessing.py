"""Data cleaning utilities."""

from __future__ import annotations

import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the flood-risk dataset while preserving transparency."""
    df = df.copy()

    # Drop exact duplicate rows to avoid repeated observations.
    df = df.drop_duplicates()

    # Convert numeric columns carefully.
    numeric_cols = [
        "rainfall_mm",
        "elevation_m",
        "slope_deg",
        "drainage_index",
        "local_flood_history",
        "road_quality",
        "proximity_to_channel",
        "impervious_surface",
        "flood_risk",
        "risk_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle impossible values.
    if "rainfall_mm" in df.columns:
        df.loc[df["rainfall_mm"] < 0, "rainfall_mm"] = None
    if "elevation_m" in df.columns:
        df.loc[df["elevation_m"] < 0, "elevation_m"] = None
    if "slope_deg" in df.columns:
        df.loc[df["slope_deg"] < 0, "slope_deg"] = None
    if "drainage_index" in df.columns:
        df.loc[(df["drainage_index"] < 0) | (df["drainage_index"] > 1), "drainage_index"] = None

    # Missing values: keep a transparent record and fill with median for numeric features.
    for col in numeric_cols:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a compact dataset summary useful for EDA and reporting."""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_distribution": df.get("flood_risk", pd.Series([0] * len(df))).value_counts().to_dict(),
    }

"""Data loading utilities for the flood-risk project.

Important note:
This project uses verified public meteorological and terrain sources, but direct
road-level flood labels are not consistently available for a full city. For that
reason, the project maintains a synthetic demonstration dataset clearly labeled as
SYNTHETIC DATA — NOT REAL-WORLD OBSERVATION.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


NOAA_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/access/services/data/v1?"
    "dataset=daily-summaries&stations={station_id}&startDate={start_date}&"
    "endDate={end_date}&format=json"
)


def download_noaa_daily_data(
    station_id: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Download daily weather data for a NOAA station."""
    url = NOAA_URL_TEMPLATE.format(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
    )
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        raise RuntimeError(
            f"NOAA download failed for station {station_id}. Status code: {response.status_code}"
        )

    data = response.json()
    df = pd.DataFrame(data)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(output, index=False)
    return df


def create_synthetic_flood_dataset(rows: int = 500) -> pd.DataFrame:
    """Generate a synthetic dataset for an academic prototype demonstration.

    This is explicitly not a real-world observation dataset. It is used only to
    test the ML pipeline and route-planning logic in a reproducible way.
    """
    rng = np.random.default_rng(42)
    records = []
    for _ in range(rows):
        rainfall = float(rng.uniform(5, 120))
        elevation = float(rng.uniform(5, 200))
        slope = float(rng.uniform(0, 30))
        drainage = float(rng.uniform(0.15, 1.0))
        local_flood_history = int(rng.integers(0, 4))
        road_quality = float(rng.uniform(0.1, 1.0))
        proximity_to_channel = float(rng.uniform(0, 1))
        impervious_surface = float(rng.uniform(0.2, 1.0))

        risk_score = (
            0.45 * rainfall / 120
            + 0.20 * (1 - drainage)
            + 0.15 * slope / 30
            + 0.15 * local_flood_history / 3
            + 0.20 * proximity_to_channel
            + 0.10 * impervious_surface
            + 0.10 * (1 - road_quality)
        )
        flood_risk = int(risk_score >= 0.52)

        record = {
            "rainfall_mm": rainfall,
            "elevation_m": elevation,
            "slope_deg": slope,
            "drainage_index": drainage,
            "local_flood_history": local_flood_history,
            "road_quality": road_quality,
            "proximity_to_channel": proximity_to_channel,
            "impervious_surface": impervious_surface,
            "flood_risk": flood_risk,
            "risk_score": round(float(risk_score), 4),
            "source_note": "SYNTHETIC DATA — NOT REAL-WORLD OBSERVATION",
        }
        records.append(record)

    return pd.DataFrame(records)


def load_project_data(
    output_path: str | Path = "data/processed/flood_demo_data.csv",
    prefer_real: bool = False,
) -> pd.DataFrame:
    """Load the project dataset.

    If a real dataset exists, it is loaded; otherwise a synthetic demonstration
    dataset is created and clearly labeled.
    """
    path = Path(output_path)
    if prefer_real and path.exists():
        return pd.read_csv(path)

    if path.exists():
        return pd.read_csv(path)

    df = create_synthetic_flood_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

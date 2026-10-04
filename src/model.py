"""Model training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def build_model_pipeline(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict]:
    """Train a Random Forest classifier and return model plus metrics."""
    feature_columns = [
        "rainfall_mm",
        "elevation_m",
        "slope_deg",
        "drainage_index",
        "local_flood_history",
        "road_quality",
        "proximity_to_channel",
        "impervious_surface",
        "rainfall_risk_flag",
        "low_drainage_flag",
        "steep_slope_flag",
        "high_history_flag",
        "road_risk_flag",
        "flood_exposure_index",
    ]

    X = df[feature_columns]
    y = df["flood_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    return model, metrics


def predict_flood_risk(model: RandomForestClassifier, sample: dict) -> tuple[int, float]:
    """Return a class prediction and probability for a single sample."""
    feature_order = [
        "rainfall_mm",
        "elevation_m",
        "slope_deg",
        "drainage_index",
        "local_flood_history",
        "road_quality",
        "proximity_to_channel",
        "impervious_surface",
        "rainfall_risk_flag",
        "low_drainage_flag",
        "steep_slope_flag",
        "high_history_flag",
        "road_risk_flag",
        "flood_exposure_index",
    ]
    vector = {name: float(sample.get(name, 0.0)) for name in feature_order}
    sample_df = pd.DataFrame([vector], columns=feature_order)

    proba = model.predict_proba(sample_df)[0]
    label = int(model.predict(sample_df)[0])
    risk_probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
    return label, risk_probability

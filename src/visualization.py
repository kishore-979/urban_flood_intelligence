"""Visualization utilities for the flood-risk project."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_class_distribution(df: pd.DataFrame, target: str = "flood_risk") -> None:
    """Plot the distribution of the target class."""
    value_counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Low risk", "High risk"], [value_counts.get(0, 0), value_counts.get(1, 0)], color=["#4CAF50", "#F44336"])
    ax.set_title("Flood risk class distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_rainfall_distribution(df: pd.DataFrame) -> None:
    """Plot rainfall distribution for the flood-risk dataset."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["rainfall_mm"], bins=20, kde=True, ax=ax)
    ax.set_title("Rainfall distribution")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: list[list[int]]) -> None:
    """Plot a simple confusion matrix using matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Low risk", "High risk"], yticklabels=["Low risk", "High risk"], ax=ax)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    plt.tight_layout()
    return fig

"""Polished Streamlit interface for the flood-risk prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import GoalBasedAgent
from src.astar import astar_search
from src.data_loader import load_project_data
from src.feature_engineering import engineer_features
from src.model import build_model_pipeline, predict_flood_risk
from src.preprocessing import clean_dataset
from src.routing import build_demo_coordinates, build_demo_road_graph, flood_aware_graph
from src.visualization import plot_class_distribution, plot_confusion_matrix, plot_rainfall_distribution


st.set_page_config(
    page_title="Urban Flood Intelligence",
    page_icon="🌊",
    layout="wide",
)

SCENARIO_VALUES = {
    "Custom": (65, 0.35, 22, 2, 0.35),
    "Routine conditions": (25, 0.75, 5, 0, 0.8),
    "Heavy rainfall": (95, 0.6, 12, 1, 0.7),
    "Poor drainage": (55, 0.2, 10, 1, 0.4),
    "High flood history": (45, 0.5, 20, 3, 0.6),
}

INPUT_KEYS = ("rainfall_input", "drainage_input", "slope_input", "history_input", "road_quality_input")


def apply_scenario() -> None:
    for key, value in zip(INPUT_KEYS, SCENARIO_VALUES[st.session_state["scenario"]]):
        st.session_state[key] = value


def mark_custom_scenario() -> None:
    st.session_state["scenario"] = "Custom"


@st.cache_data
def get_project_data() -> tuple[pd.DataFrame, object, dict]:
    df = load_project_data()
    df = clean_dataset(df)
    df = engineer_features(df)
    model, metrics = build_model_pipeline(df)
    return df, model, metrics


def build_sample(rainfall: float, drainage: float, slope: float, history: int, road_quality: float) -> dict:
    return {
        "rainfall_mm": rainfall,
        "elevation_m": 40.0,
        "slope_deg": slope,
        "drainage_index": drainage,
        "local_flood_history": history,
        "road_quality": road_quality,
        "proximity_to_channel": 0.7,
        "impervious_surface": 0.8,
        "rainfall_risk_flag": int(rainfall > 60),
        "low_drainage_flag": int(drainage < 0.45),
        "steep_slope_flag": int(slope > 18),
        "high_history_flag": int(history > 1),
        "road_risk_flag": int(road_quality < 0.5),
        "flood_exposure_index": 0.4 * rainfall / 120 + 0.2 * (1 - drainage) + 0.2 * slope / 30 + 0.2 * 0.7,
    }


def build_demo_route_view() -> pd.DataFrame:
    coordinates = {
        "A": (29.7604, -95.3698),
        "B": (29.7660, -95.3650),
        "C": (29.7540, -95.3770),
        "D": (29.7705, -95.3700),
        "E": (29.7770, -95.3590),
        "F": (29.7480, -95.3850),
        "G": (29.7850, -95.3470),
    }
    route_points = []
    for node, (lat, lon) in coordinates.items():
        route_points.append({"node": node, "lat": lat, "lon": lon, "risk": 0.2 + (0.1 * (ord(node) % 5))})
    return pd.DataFrame(route_points)


def render_dashboard() -> None:
    st.title("🌊 Urban Flood Intelligence")
    st.caption(
        "Flood-risk prediction and flood-aware route planning · Academic prototype with a clearly labeled synthetic risk layer."
    )

    df, model, metrics = get_project_data()

    default_values = SCENARIO_VALUES["Custom"]
    for key, value in zip(INPUT_KEYS, default_values):
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.header("Scenario inputs")
        st.caption("Choose a preset or adjust any value. Changes update the flood-risk prediction.")
        st.selectbox(
            "Starting scenario",
            tuple(SCENARIO_VALUES),
            key="scenario",
            on_change=apply_scenario,
            help="Presets are illustrative examples, not observed weather events.",
        )
        rainfall = st.slider(
            "Rainfall (mm)", 0, 120, key=INPUT_KEYS[0], on_change=mark_custom_scenario,
            help="Illustrative rainfall amount used as one model input.",
        )
        drainage = st.slider(
            "Drainage capacity", 0.0, 1.0, key=INPUT_KEYS[1], on_change=mark_custom_scenario,
            help="Demo scale from 0 (low capacity) to 1 (high capacity).",
        )
        slope = st.slider(
            "Terrain slope (degrees)", 0, 30, key=INPUT_KEYS[2], on_change=mark_custom_scenario,
            help="Illustrative terrain slope in degrees.",
        )
        history = st.slider(
            "Previous flood events", 0, 3, key=INPUT_KEYS[3], on_change=mark_custom_scenario,
            help="Demo count of prior local events; this is not a live local record.",
        )
        road_quality = st.slider(
            "Road quality", 0.0, 1.0, key=INPUT_KEYS[4], on_change=mark_custom_scenario,
            help="Demo scale from 0 (poor) to 1 (good).",
        )
        st.markdown("---")
        st.caption("Academic demonstration · Houston / Harris County context")
        st.caption("The road graph is a separate illustrative example.")

    sample = build_sample(rainfall, drainage, slope, history, road_quality)
    label, prob = predict_flood_risk(model, sample)
    risk_level = "HIGH FLOOD RISK" if label == 1 else "LOW FLOOD RISK"
    risk_color = "#ff4b4b" if label == 1 else "#4cc9f0"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset rows", len(df))
    with col2:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col3:
        st.metric("F1 score", f"{metrics['f1']:.3f}")

    tabs = st.tabs(["Overview", "Flood risk", "Routes", "Evaluation"])

    with tabs[0]:
        st.subheader("Your project, in four steps")
        guide_columns = st.columns(4)
        for column, step, detail in zip(
            guide_columns,
            ("1 · Set conditions", "2 · Review risk", "3 · Compare routes", "4 · Check evaluation"),
            ("Use the sidebar presets or sliders.", "See the prediction and its input signals.", "Compare shortest and flood-penalized paths.", "Inspect model scores and class results."),
        ):
            column.markdown(f"**{step}**")
            column.caption(detail)

        overview_col1, overview_col2 = st.columns([1.3, 1])
        with overview_col1:
            st.metric(
                "Model prediction",
                risk_level,
                help="The classifier's predicted class for the current sidebar inputs.",
            )
            st.metric(
                "Estimated high-risk probability",
                f"{prob:.1%}",
                help="The model's estimated probability for the high-risk class; it is not a real-time forecast.",
            )
            st.progress(float(prob), text="Estimated high-risk probability")
            st.info("This academic prototype uses a synthetic demonstration risk layer. Do not use it for real-world travel or emergency decisions.")
        with overview_col2:
            route_df = build_demo_route_view()
            st.caption("Illustrative sample locations · not live flood observations")
            st.map(route_df[["lat", "lon"]], use_container_width=True)

        st.subheader("Sample dataset")
        st.caption("Preview of the data used by the prototype; the displayed rows do not change with the scenario controls.")
        st.dataframe(df.head(10), width="stretch")

    with tabs[1]:
        st.subheader("Current scenario conditions")
        condition_columns = st.columns(3)
        conditions = [
            ("Rainfall", f"{rainfall} mm"),
            ("Drainage capacity", f"{drainage:.2f} / 1.00"),
            ("Terrain slope", f"{slope}°"),
            ("Flood history", f"{history} events"),
            ("Road quality", f"{road_quality:.2f} / 1.00"),
        ]
        for column, (label_text, value) in zip(condition_columns, conditions[:3]):
            column.metric(label_text, value)
        condition_columns = st.columns(3)
        for column, (label_text, value) in zip(condition_columns, conditions[3:]):
            column.metric(label_text, value)

        st.subheader("Input signals to review")
        st.caption("These simple threshold cues help explain the demo inputs; they are not feature-importance scores from the model.")
        reasons = []
        if rainfall > 60:
            reasons.append("Rainfall is above the demo threshold (60 mm).")
        if drainage < 0.45:
            reasons.append("Drainage capacity is below the demo threshold (0.45).")
        if slope > 18:
            reasons.append("Terrain slope is above the demo threshold (18°).")
        if history > 1:
            reasons.append("Flood history is above the demo threshold (1 event).")
        if road_quality < 0.5:
            reasons.append("Road quality is below the demo threshold (0.50).")
        if reasons:
            for reason in reasons:
                st.markdown(f"- {reason}")
        else:
            st.success("None of the displayed demo thresholds were crossed. The model prediction may still reflect other features.")

        st.subheader("Example training-data values")
        st.caption("These rows show dataset feature values and are independent of your current scenario.")
        st.bar_chart(df[["rainfall_mm", "slope_deg", "drainage_index"]].head(20))

    with tabs[2]:
        route_graph = build_demo_road_graph()
        coordinates = build_demo_coordinates()
        risk_map = {"A": 0.2, "B": 0.4, "C": 0.7, "D": 0.3, "E": 0.5, "F": 0.8, "G": 0.2}
        adjusted_graph = flood_aware_graph(route_graph, risk_map)
        route_nodes = list(route_graph)
        route_start_col, route_goal_col = st.columns(2)
        with route_start_col:
            route_start = st.selectbox("Start node", route_nodes, index=0, help="Choose a node in the illustrative road graph.")
        with route_goal_col:
            route_goal = st.selectbox("Destination node", route_nodes, index=6, help="Choose a destination node in the illustrative road graph.")

        st.subheader("Route comparison")
        st.caption("Flood-aware cost combines graph distance with a fixed penalty for risk at the next node. These demo routes do not use the sidebar flood-risk prediction.")
        if route_start == route_goal:
            st.warning("Choose different start and destination nodes to compare routes.")
        else:
            shortest_path, shortest_cost = astar_search(route_graph, route_start, route_goal, coordinates)
            safe_path, safe_cost = astar_search(adjusted_graph, route_start, route_goal, coordinates)
            route_col1, route_col2 = st.columns(2)
            with route_col1:
                st.metric("Shortest path", f"{len(shortest_path) - 1} links", help="Path with distance-only edge costs.")
                st.caption(" → ".join(shortest_path))
                st.caption(f"Distance cost: {shortest_cost:.2f}")
            with route_col2:
                st.metric("Flood-aware path", f"{len(safe_path) - 1} links", help="Path with distance plus the demonstration flood-risk penalty.")
                st.caption(" → ".join(safe_path))
                st.caption(f"Combined cost: {safe_cost:.2f}")

            agent = GoalBasedAgent(route_start, route_goal, risk_map)
            st.markdown("**Agent status**")
            st.info(agent.decide_action(route_start, route_goal).replace("_", " ").capitalize())

        route_map_df = pd.DataFrame([
            {"node": "A", "lat": 29.7604, "lon": -95.3698},
            {"node": "B", "lat": 29.7660, "lon": -95.3650},
            {"node": "C", "lat": 29.7540, "lon": -95.3770},
            {"node": "D", "lat": 29.7705, "lon": -95.3700},
            {"node": "E", "lat": 29.7770, "lon": -95.3590},
            {"node": "F", "lat": 29.7480, "lon": -95.3850},
            {"node": "G", "lat": 29.7850, "lon": -95.3470},
        ])
        st.caption("Map markers show illustrative graph nodes, not a navigable street map or live road conditions.")
        st.map(route_map_df[["lat", "lon"]], use_container_width=True)

    with tabs[3]:
        st.subheader("Model evaluation")
        st.caption("Scores summarize this prototype's model evaluation; they do not indicate real-world forecasting performance.")
        for labels, keys in (
            (("Accuracy", "Precision"), ("accuracy", "precision")),
            (("Recall", "F1 score"), ("recall", "f1")),
        ):
            metric_columns = st.columns(2)
            for column, label_text, key in zip(metric_columns, labels, keys):
                column.metric(label_text, f"{metrics[key]:.3f}")

        st.caption("Confusion matrix · rows are actual classes; columns are predicted classes.")
        confusion = metrics["confusion_matrix"]
        st.dataframe(
            pd.DataFrame(
                confusion,
                index=["Actual: low risk", "Actual: high risk"],
                columns=["Predicted: low risk", "Predicted: high risk"],
            ),
            width="stretch",
        )
        confusion_fig = plot_confusion_matrix(metrics["confusion_matrix"])
        st.pyplot(confusion_fig)

        with st.expander("Explore dataset charts"):
            st.subheader("Class distribution")
            class_fig = plot_class_distribution(df)
            st.pyplot(class_fig)

            st.subheader("Rainfall distribution")
            rainfall_fig = plot_rainfall_distribution(df)
            st.pyplot(rainfall_fig)


if __name__ == "__main__":
    render_dashboard()

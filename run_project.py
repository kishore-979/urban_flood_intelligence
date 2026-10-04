"""Entry point to run the project demonstration."""

from __future__ import annotations

from src.agent import GoalBasedAgent
from src.astar import astar_search
from src.data_loader import load_project_data
from src.feature_engineering import engineer_features
from src.model import build_model_pipeline, predict_flood_risk
from src.preprocessing import clean_dataset
from src.routing import build_demo_coordinates, build_demo_road_graph, flood_aware_graph


def main() -> None:
    print("INTELLIGENT URBAN FLOOD RISK PROTOTYPE")
    print("====================================")
    print("Using verified public sources where possible and a synthetic demonstration risk layer where no full real-world road-level labels exist.")

    df = load_project_data()
    df = clean_dataset(df)
    df = engineer_features(df)
    model, metrics = build_model_pipeline(df)

    sample = {
        "rainfall_mm": 75,
        "elevation_m": 30,
        "slope_deg": 22,
        "drainage_index": 0.35,
        "local_flood_history": 2,
        "road_quality": 0.4,
        "proximity_to_channel": 0.8,
        "impervious_surface": 0.85,
        "rainfall_risk_flag": 1,
        "low_drainage_flag": 1,
        "steep_slope_flag": 1,
        "high_history_flag": 1,
        "road_risk_flag": 1,
        "flood_exposure_index": 0.65,
    }
    label, probability = predict_flood_risk(model, sample)
    print(f"Flood risk prediction: {label} with probability {probability:.3f}")
    print(f"Model metrics: {metrics}")

    graph = build_demo_road_graph()
    coordinates = build_demo_coordinates()
    risk_map = {"A": 0.2, "B": 0.4, "C": 0.7, "D": 0.3, "E": 0.5, "F": 0.8, "G": 0.2}
    short_path, short_cost = astar_search(graph, "A", "G", coordinates)
    adjusted_graph = flood_aware_graph(graph, risk_map)
    safe_path, safe_cost = astar_search(adjusted_graph, "A", "G", coordinates)

    print(f"Shortest route: {short_path} cost={short_cost:.2f}")
    print(f"Flood-aware route: {safe_path} cost={safe_cost:.2f}")

    agent = GoalBasedAgent("A", "G", risk_map)
    print(f"Agent recommendation: {agent.decide_action('C', 'G')}")


if __name__ == "__main__":
    main()

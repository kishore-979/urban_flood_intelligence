"""Road graph and route planning utilities."""

from __future__ import annotations

from typing import Dict, List


def build_demo_road_graph() -> dict[str, dict[str, float]]:
    """Construct a small road network for prototype demonstration."""
    return {
        "A": {"B": 4.0, "C": 8.0},
        "B": {"A": 4.0, "D": 5.0, "E": 6.0},
        "C": {"A": 8.0, "D": 3.0, "F": 7.0},
        "D": {"B": 5.0, "C": 3.0, "G": 6.0},
        "E": {"B": 6.0, "G": 4.0},
        "F": {"C": 7.0, "G": 5.0},
        "G": {"D": 6.0, "E": 4.0, "F": 5.0},
    }


def build_demo_coordinates() -> dict[str, tuple[float, float]]:
    return {
        "A": (0.0, 0.0),
        "B": (1.0, 1.0),
        "C": (1.0, -1.0),
        "D": (2.0, 0.0),
        "E": (3.0, 2.0),
        "F": (3.0, -2.0),
        "G": (4.0, 0.0),
    }


def edge_cost(distance: float, flood_risk: float, risk_penalty: float = 4.5) -> float:
    """Route cost with a flood-risk penalty. This is a transparent prototype formula."""
    return distance + (flood_risk * risk_penalty)


def flood_aware_graph(graph: dict[str, dict[str, float]], risk_map: dict[str, float]) -> dict[str, dict[str, float]]:
    """Convert raw distances into cost with risk penalties."""
    adjusted = {}
    for node, neighbors in graph.items():
        adjusted[node] = {}
        for neighbor, distance in neighbors.items():
            risk = risk_map.get(neighbor, 0.0)
            adjusted[node][neighbor] = edge_cost(distance, risk)
    return adjusted


def shortest_route(graph: dict[str, dict[str, float]], start: str, goal: str) -> list[str]:
    """A basic shortest-path route using the graph weights directly."""
    if start == goal:
        return [start]
    queue = [(start, [start])]
    visited = set()
    while queue:
        node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, distance in graph.get(node, {}).items():
            if neighbor == goal:
                return path + [neighbor]
            queue.append((neighbor, path + [neighbor]))
    return []

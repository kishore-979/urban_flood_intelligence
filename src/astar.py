"""A* search implementation for flood-aware route planning."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def heuristic(a: str, b: str, coordinates: dict[str, tuple[float, float]]) -> float:
    """Simple Euclidean heuristic for the route planner."""
    ax, ay = coordinates.get(a, (0.0, 0.0))
    bx, by = coordinates.get(b, (0.0, 0.0))
    return math.hypot(ax - bx, ay - by)


def astar_search(
    graph: dict[str, dict[str, float]],
    start: str,
    goal: str,
    coordinates: dict[str, tuple[float, float]],
) -> tuple[list[str], float]:
    """Return the best path and cost using A* search."""
    open_set: set[str] = {start}
    came_from: dict[str, str | None] = {start: None}
    g_score: dict[str, float] = {start: 0.0}
    f_score: dict[str, float] = {start: heuristic(start, goal, coordinates)}

    while open_set:
        current = min(open_set, key=lambda node: f_score.get(node, float("inf")))
        if current == goal:
            path: list[str] = []
            cursor: str | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = came_from.get(cursor)
            return list(reversed(path)), g_score[current]

        open_set.remove(current)
        for neighbor, edge_cost in graph.get(current, {}).items():
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal, coordinates)
                open_set.add(neighbor)

    return [], float("inf")

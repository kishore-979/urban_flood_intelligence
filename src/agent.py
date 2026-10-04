"""Goal-based intelligent agent for route planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentState:
    current_location: str
    destination: str
    current_risk: float
    flood_aware_route: list[str] | None = None


class GoalBasedAgent:
    """A simple intelligent agent that prefers lower-risk routes."""

    def __init__(self, start: str, destination: str, risk_map: dict[str, float]):
        self.start = start
        self.destination = destination
        self.risk_map = risk_map

    def perceive(self, current_location: str) -> dict:
        return {
            "location": current_location,
            "flood_risk": self.risk_map.get(current_location, 0.0),
        }

    def decide_action(self, current_location: str, destination: str) -> str:
        risk = self.risk_map.get(current_location, 0.0)
        if risk > 0.7:
            return "avoid_route"
        if current_location == destination:
            return "arrive"
        return "move_towards_destination"

    def choose_route(self, possible_routes: list[list[str]], risk_map: dict[str, float]) -> list[str]:
        scored_routes = []
        for route in possible_routes:
            route_risk = sum(risk_map.get(node, 0.0) for node in route)
            scored_routes.append((route_risk, route))
        scored_routes.sort(key=lambda item: item[0])
        return scored_routes[0][1]

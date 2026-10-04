"""Backward chaining for goal verification."""

from __future__ import annotations

from .fol_engine import FOLKnowledgeBase, FOLRule


def build_default_rules() -> list[FOLRule]:
    return [
        FOLRule(
            name="R1",
            premises=["HighRainfall(x)", "LowDrainage(x)"],
            conclusion="HighFloodRisk(x)",
        ),
        FOLRule(
            name="R2",
            premises=["HighFloodRisk(x)"],
            conclusion="UnsafeRoad(x)",
        ),
        FOLRule(
            name="R3",
            premises=["SafeRoad(x)"],
            conclusion="SafeRoute(Home, Hospital)",
        ),
    ]


def run_backward_chain(goal: str, facts: set[str] | None = None) -> tuple[bool, list[str]]:
    """Check whether a target goal can be inferred by backward chaining."""
    kb = FOLKnowledgeBase()
    if facts is None:
        facts = {
            "HighRainfall(ZoneA)",
            "LowDrainage(ZoneA)",
            "SafeRoad(R1)",
            "SafeRoad(R2)",
        }

    for fact in facts:
        kb.add_fact(fact)
    for rule in build_default_rules():
        kb.add_rule(rule)

    trace: list[str] = []
    result = kb.backward_chain(goal, trace)
    return result, trace

"""Forward chaining engine for flood-risk inference."""

from __future__ import annotations

from .fol_engine import FOLKnowledgeBase, FOLRule


def build_default_flood_rules() -> list[FOLRule]:
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
            premises=["UnsafeRoad(x)"],
            conclusion="AvoidRoad(x)",
        ),
        FOLRule(
            name="HighRainfall(x)" ,
            premises=[],
            conclusion="HighRainfall(x)",
        ),
    ]


def run_forward_chain(facts: set[str] | None = None) -> tuple[set[str], list[str]]:
    """Run the engine and return the deduced facts and a trace."""
    kb = FOLKnowledgeBase()
    if facts is None:
        facts = {
            "HighRainfall(ZoneA)",
            "LowDrainage(ZoneA)",
            "RoadStatus(R7, unsafe)",
        }

    for fact in facts:
        kb.add_fact(fact)

    for rule in build_default_flood_rules():
        if rule.name == "R1":
            pass
        kb.add_rule(rule)

    final_facts = kb.forward_chain()
    return final_facts, list(sorted(final_facts))

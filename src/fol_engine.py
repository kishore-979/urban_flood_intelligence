"""Knowledge base and first-order logic utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FOLRule:
    """A simple Horn-clause-like rule: premises -> conclusion."""
    name: str
    premises: list[str]
    conclusion: str


class FOLKnowledgeBase:
    """Small rule engine for explainable flood-risk reasoning."""

    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_fact(self, fact: str) -> None:
        self.facts.add(fact)

    def add_rule(self, rule: FOLRule) -> None:
        self.rules.append(rule)

    def forward_chain(self) -> set[str]:
        """Apply rules repeatedly until nothing new can be inferred."""
        new_facts = set(self.facts)
        while True:
            derived = False
            for rule in self.rules:
                if all(premise in new_facts for premise in rule.premises):
                    if rule.conclusion not in new_facts:
                        new_facts.add(rule.conclusion)
                        derived = True
            if not derived:
                break
        self.facts = new_facts
        return set(self.facts)

    def backward_chain(self, goal: str, trace: list[str] | None = None) -> bool:
        """Check whether a goal can be satisfied under the current knowledge base."""
        if trace is None:
            trace = []

        if goal in self.facts:
            trace.append(f"Goal satisfied directly: {goal}")
            return True

        for rule in self.rules:
            if rule.conclusion == goal:
                trace.append(f"Checking rule: {rule.name}")
                if all(self.backward_chain(premise, trace) for premise in rule.premises):
                    trace.append(f"Goal proved via {rule.name}: {goal}")
                    return True
                trace.append(f"Rule {rule.name} failed for {goal}")
        trace.append(f"Goal could not be proved: {goal}")
        return False

from __future__ import annotations

from typing import List, Set


class FOLKnowledgeBase:
    def __init__(self):
        self.facts: Set[str] = set()

    def assert_fact(self, fact: str) -> None:
        self.facts.add(fact)

    def query(self, fact: str) -> bool:
        return fact in self.facts

    def infer(self, rule: str, facts: List[str]) -> bool:
        return rule in self.facts or fact_exists_in_list(rule, facts)


def fact_exists_in_list(rule: str, facts: List[str]) -> bool:
    return any(rule == fact for fact in facts)

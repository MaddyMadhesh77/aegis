from __future__ import annotations

from typing import Iterable, Set


def normalize_literal(literal: str) -> str:
    return literal.strip()


def resolve_clauses(clause_a: Iterable[str], clause_b: Iterable[str]) -> Set[str]:
    left = set(normalize_literal(lit) for lit in clause_a)
    right = set(normalize_literal(lit) for lit in clause_b)

    result: Set[str] = set()
    for literal in left:
        complement = literal[1:] if literal.startswith("~") else f"~{literal}"
        if complement in right:
            result = (left | right) - {literal, complement}
            return result
    for literal in right:
        complement = literal[1:] if literal.startswith("~") else f"~{literal}"
        if complement in left:
            result = (left | right) - {literal, complement}
            return result
    return set()

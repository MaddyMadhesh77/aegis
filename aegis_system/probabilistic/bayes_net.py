from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


class BayesNet:
    """Minimal exact-inference Bayesian network implementation."""

    def __init__(self, variables: Iterable[str], cpts: Dict[str, Any]):
        self.variables = list(variables)
        self.cpts = cpts

    def infer(self, variable: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[Any, float]:
        if evidence is None:
            evidence = {}

        values = [True, False]
        probs: Dict[Any, float] = {}

        for value in values:
            total = 0.0
            for assignment in self._enumerate_assignments():
                if not self._matches_evidence(assignment, evidence):
                    continue
                if assignment.get(variable) == value:
                    total += self._probability_of_assignment(assignment)
            probs[value] = total

        total = sum(probs.values())
        if total == 0:
            return {True: 0.5, False: 0.5}
        return {key: value / total for key, value in probs.items()}

    def _enumerate_assignments(self) -> List[Dict[str, Any]]:
        assignments: List[Dict[str, Any]] = [dict()]
        for var in self.variables:
            next_assignments: List[Dict[str, Any]] = []
            for state in [True, False]:
                for assignment in assignments:
                    new_assignment = dict(assignment)
                    new_assignment[var] = state
                    next_assignments.append(new_assignment)
            assignments = next_assignments
        return assignments

    def _matches_evidence(self, assignment: Dict[str, Any], evidence: Dict[str, Any]) -> bool:
        for var, value in evidence.items():
            if assignment.get(var) != value:
                return False
        return True

    def _probability_of_assignment(self, assignment: Dict[str, Any]) -> float:
        prob = 1.0
        for var in self.variables:
            cpt = self.cpts.get(var)
            if cpt is None:
                continue
            parents = defaultdict(dict)
            if not cpt:
                prob *= cpt.get((), {}).get(assignment[var], 0.0)
                continue
            for parent_key, cond in cpt.items():
                if isinstance(parent_key, tuple):
                    if all(assignment.get(p) == v for p, v in zip(parent_key, parent_key)):
                        pass
                if isinstance(parent_key, str):
                    if assignment.get(parent_key) == None:
                        continue
                if parent_key == ():
                    prob *= cond.get(assignment[var], 0.0)
                    continue
                if isinstance(parent_key, tuple):
                    parent_values = dict(zip(parent_key, [assignment.get(p) for p in parent_key]))
                    value = cond.get(parent_values, {}).get(assignment[var], 0.0)
                    prob *= value
                    break
                if assignment.get(parent_key) is not None:
                    value = cond.get(assignment[parent_key], {}).get(assignment[var], 0.0)
                    prob *= value
            
        return prob

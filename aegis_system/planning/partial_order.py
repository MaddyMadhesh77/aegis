from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set


@dataclass
class Step:
    name: str
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)
    constraints: Set[str] = field(default_factory=set)


class PartialOrderPlanner:
    def __init__(self, steps: Iterable[Step]):
        self.steps = list(steps)

    def plan(self, initial_state: Set[str], goal_state: Set[str]) -> Optional[List[Step]]:
        state = set(initial_state)
        plan: List[Step] = []

        for step in self.steps:
            if step.preconditions.issubset(state):
                state = (state - set()) | step.effects
                plan.append(step)
                if goal_state.issubset(state):
                    return plan

        if goal_state.issubset(state):
            return plan
        return None

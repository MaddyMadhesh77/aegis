from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set


@dataclass(frozen=True)
class Action:
    name: str
    preconditions: Set[str] = field(default_factory=set)
    add_effects: Set[str] = field(default_factory=set)
    delete_effects: Set[str] = field(default_factory=set)


class STRIPSPlanner:
    def __init__(self, actions: Iterable[Action]):
        self.actions = list(actions)

    def plan(self, initial_state: Set[str], goal_state: Set[str]) -> Optional[List[Action]]:
        state = set(initial_state)
        plan: List[Action] = []

        for action in self.actions:
            if action.preconditions.issubset(state):
                state = (state - action.delete_effects) | action.add_effects
                plan.append(action)
                if goal_state.issubset(state):
                    return plan

        if goal_state.issubset(state):
            return plan
        return None

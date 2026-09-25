"""Partial-order planning (the textbook POP algorithm), as recursive DFS with backtracking.

A partial plan holds steps, ordering constraints a < b, causal links (a, p, b)
meaning "a achieves precondition p of b", and open preconditions. Start adds the
initial state; Finish requires the goal.

refine(plan):
    if a step threatens a causal link (deletes p of a link a -p-> b and can fall between a and b):
        branch on demotion (step < a) and promotion (b < step)
    elif no open preconditions: return plan
    else pick the open precondition with the fewest achievers and branch on
         each existing step that adds it, then each new action that adds it

Works on the ground positive STRIPS problems built by strips_planner.ground,
so negative preconditions arrive as Not<p> atoms and are protected by causal
links like any other precondition.

`layers()` turns the orderings into parallel stages: a topological layering in
which every step goes in the earliest layer its predecessors allow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from ..core.logic import Atom
from ..core.types import Plan
from .strips_planner import Action, Problem

START, FINISH = 0, 1
Link = Tuple[int, Atom, int]


@dataclass
class PartialPlan:
    steps: Dict[int, Optional[Action]]                  # id -> action (None for Start and Finish)
    orderings: Set[Tuple[int, int]]
    links: Set[Link]
    open: List[Tuple[Atom, int]]                        # (precondition, step that needs it)
    next_id: int = 2

    def copy(self) -> "PartialPlan":
        return PartialPlan(dict(self.steps), set(self.orderings), set(self.links), list(self.open), self.next_id)


@dataclass
class POPStats:
    nodes: int = 0
    backtracks: int = 0


@dataclass
class POPResult:
    plan: Optional[PartialPlan]
    stats: POPStats = field(default_factory=POPStats)

    @property
    def solved(self) -> bool:
        return self.plan is not None


class PartialOrderPlanner:
    def __init__(self, problem: Problem, max_steps: int = 12, max_nodes: int = 50_000):
        self.problem = problem
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.achievers: Dict[Atom, List[Action]] = {}
        for a in sorted(problem.actions, key=lambda a: (problem.step_cost(a), str(a))):
            for p in a.add:
                self.achievers.setdefault(p, []).append(a)

    # ----------------------------------------------------------------- public

    def plan(self) -> POPResult:
        p = self.problem
        initial = PartialPlan({START: None, FINISH: None}, {(START, FINISH)}, set(),
                              [(g, FINISH) for g in sorted(p.goal, key=str)])
        stats = POPStats()
        result = self._refine(initial, stats)
        return POPResult(result, stats)

    # ----------------------------------------------------------------- step helpers

    def adds(self, plan: PartialPlan, sid: int) -> FrozenSet[Atom]:
        return self.problem.init if sid == START else (plan.steps[sid].add if plan.steps[sid] else frozenset())

    def deletes(self, plan: PartialPlan, sid: int) -> FrozenSet[Atom]:
        return plan.steps[sid].delete if plan.steps[sid] else frozenset()

    @staticmethod
    def _before(orderings: Set[Tuple[int, int]], a: int, b: int) -> bool:
        """True if a < b follows from the orderings (DFS over the ordering graph)."""
        succ: Dict[int, List[int]] = {}
        for x, y in orderings:
            succ.setdefault(x, []).append(y)
        stack, seen = [a], {a}
        while stack:
            for nxt in succ.get(stack.pop(), ()):
                if nxt == b:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def _add_ordering(self, plan: PartialPlan, a: int, b: int) -> bool:
        """Add a < b unless it creates a cycle."""
        if a == b or self._before(plan.orderings, b, a):
            return False
        plan.orderings.add((a, b))
        return True

    def _threats(self, plan: PartialPlan) -> List[Tuple[int, Link]]:
        out = []
        for link in sorted(plan.links, key=lambda l: (l[0], str(l[1]), l[2])):
            a, p, b = link
            for s in sorted(plan.steps):
                if s in (a, b) or p not in self.deletes(plan, s):
                    continue
                if self._before(plan.orderings, s, a) or self._before(plan.orderings, b, s):
                    continue
                out.append((s, link))
        return out

    # ----------------------------------------------------------------- search

    def _refine(self, plan: PartialPlan, stats: POPStats) -> Optional[PartialPlan]:
        stats.nodes += 1
        if stats.nodes > self.max_nodes:
            return None

        threats = self._threats(plan)
        if threats:
            s, (a, _, b) = threats[0]
            for before, after in ((s, a), (b, s)):  # demotion, then promotion
                child = plan.copy()
                if self._add_ordering(child, before, after):
                    found = self._refine(child, stats)
                    if found is not None:
                        return found
            stats.backtracks += 1
            return None

        if not plan.open:
            return plan

        # most constrained open precondition first
        def n_options(item: Tuple[Atom, int]) -> int:
            p, _ = item
            existing = sum(1 for s in plan.steps if p in self.adds(plan, s))
            return existing + len(self.achievers.get(p, ()))

        goal_item = min(plan.open, key=lambda item: (n_options(item), str(item[0]), item[1]))
        p, consumer = goal_item
        rest = [item for item in plan.open if item != goal_item]

        # 1. reuse an existing step (least commitment keeps the plan short and parallel)
        for s in sorted(plan.steps):
            if s == FINISH or s == consumer or p not in self.adds(plan, s):
                continue
            if self._before(plan.orderings, consumer, s):
                continue
            child = plan.copy()
            child.open = rest
            if not self._add_ordering(child, s, consumer):
                continue
            child.links.add((s, p, consumer))
            found = self._refine(child, stats)
            if found is not None:
                return found

        # 2. add a new step
        if len(plan.steps) - 2 < self.max_steps:
            for action in self.achievers.get(p, ()):
                child = plan.copy()
                sid = child.next_id
                child.next_id += 1
                child.steps[sid] = action
                child.orderings |= {(START, sid), (sid, FINISH)}
                if not self._add_ordering(child, sid, consumer):
                    continue
                child.links.add((sid, p, consumer))
                child.open = rest + [(q, sid) for q in sorted(action.pre, key=str)]
                found = self._refine(child, stats)
                if found is not None:
                    return found

        stats.backtracks += 1
        return None


# --------------------------------------------------------------------------- output


def layers(plan: PartialPlan) -> List[List[Action]]:
    """Parallel stages: every step in the earliest layer its orderings allow."""
    g = nx.DiGraph()
    real = [s for s in plan.steps if s not in (START, FINISH)]
    g.add_nodes_from(real)
    g.add_edges_from((a, b) for a, b in nx.transitive_closure_dag(_order_graph(plan)).edges()
                     if a in g and b in g)
    return [sorted((plan.steps[s] for s in gen), key=str) for gen in nx.topological_generations(g)]


def _order_graph(plan: PartialPlan) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(plan.steps)
    g.add_edges_from(plan.orderings)
    return g


def linearize(plan: PartialPlan) -> List[Action]:
    """One total order consistent with the plan."""
    return [a for layer in layers(plan) for a in layer]


def to_plan(plan: PartialPlan, trace_id: Optional[str] = None) -> Plan:
    """The core Plan record: layered steps plus orderings and causal links between actions."""
    name = {s: ("Start" if s == START else "Finish" if s == FINISH else str(plan.steps[s])) for s in plan.steps}
    return Plan(
        steps=layers(plan),
        orderings={(name[a], name[b]) for a, b in plan.orderings},
        causal_links={(name[a], p, name[b]) for a, p, b in plan.links},
        trace_id=trace_id,
    )


def format_layers(stages: List[List[Action]]) -> str:
    return "[" + ", ".join("{" + ", ".join(str(a) for a in layer) + "}" for layer in stages) + "]"

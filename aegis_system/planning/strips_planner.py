"""STRIPS planning by state-space search.

Action schemas have parameters and are grounded over the scenario's objects.
Preconditions may be negative (closed world: ~Isolated(h) holds when Isolated(h)
is not in the state). Grounding compiles each negated atom p into a positive
atom Not<p> that the actions keep up to date, so every planner downstream works
on plain positive STRIPS:

    Isolate(h):  pre Compromised(h) & ~Isolated(h)   add Isolated(h)
    grounded:    pre {Compromised(H1), NotIsolated(H1)}  add {Isolated(H1)}  del {NotIsolated(H1)}

Planners
    progression  A* forward over states (frozensets of Atom)
    regression   A* backward over goal sets: regress(g, a) = (g - add(a)) | pre(a)

Heuristics come from the delete relaxation, computed by a fixpoint over the
actions: h_max (cost of the most expensive goal atom, admissible) and h_add
(sum over goal atoms, more informed but NOT admissible because it counts shared
sub-goals twice).

Action costs blend money and time: step_cost = (1 - urgency) * cost + urgency * duration.
Urgency comes from the predicted time-to-compromise (see urgency_from_ttc).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from ..core.logic import And, Atom, Const, Formula, Not, Term, Var, parse
from ..reasoning.unifier import substitute

INF = float("inf")
State = FrozenSet[Atom]


# --------------------------------------------------------------------------- literals


def literals(text: Union[str, Formula, None]) -> Tuple[List[Atom], List[Atom]]:
    """Split a conjunction of literals into (positive atoms, negated atoms)."""
    if text is None or text == "":
        return [], []
    f = parse(text) if isinstance(text, str) else text
    pos: List[Atom] = []
    neg: List[Atom] = []

    def walk(node: Formula) -> None:
        if isinstance(node, And):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, Atom):
            pos.append(node)
        elif isinstance(node, Not) and isinstance(node.arg, Atom):
            neg.append(node.arg)
        else:
            raise ValueError(f"Expected a conjunction of literals, got {node}")

    walk(f)
    return pos, neg


def negated(atom: Atom) -> Atom:
    """The positive atom that stands for ~atom after compilation."""
    return Atom("Not" + atom.pred, atom.args)


def atoms(text: str) -> FrozenSet[Atom]:
    pos, neg = literals(text)
    if neg:
        raise ValueError(f"Expected positive atoms only: {text}")
    return frozenset(pos)


# --------------------------------------------------------------------------- actions


@dataclass(frozen=True)
class ActionSchema:
    name: str
    params: Tuple[Var, ...]
    pre_pos: Tuple[Atom, ...] = ()
    pre_neg: Tuple[Atom, ...] = ()
    add: Tuple[Atom, ...] = ()
    delete: Tuple[Atom, ...] = ()
    cost: float = 1.0
    duration: float = 1.0

    @classmethod
    def parse(cls, head: str, pre: str = "", add: str = "", delete: str = "", cost: float = 1.0,
              duration: float = 1.0) -> "ActionSchema":
        """ActionSchema.parse("Isolate(h)", pre="Compromised(h) & ~Isolated(h)", add="Isolated(h)")."""
        h = parse(head)
        if not isinstance(h, Atom) or not all(isinstance(a, Var) for a in h.args):
            raise ValueError(f"Action head must look like Name(var, ...): {head}")
        pre_pos, pre_neg = literals(pre)
        add_pos, add_neg = literals(add)
        del_pos, del_neg = literals(delete)
        if add_neg or del_neg:
            raise ValueError("Effects are lists of atoms; use `delete` instead of negated effects")
        return cls(h.pred, tuple(h.args), tuple(pre_pos), tuple(pre_neg), tuple(add_pos), tuple(del_pos),
                   cost, duration)


@dataclass(frozen=True)
class Action:
    """A ground action. `pre` is positive only (negative preconditions are compiled)."""

    name: str
    args: Tuple[Term, ...]
    pre: FrozenSet[Atom]
    add: FrozenSet[Atom]
    delete: FrozenSet[Atom]
    cost: float = 1.0
    duration: float = 1.0

    def applicable(self, state: State) -> bool:
        return self.pre <= state

    def apply(self, state: State) -> State:
        return (state - self.delete) | self.add

    @property
    def atom(self) -> Atom:
        return Atom(self.name, self.args)

    def __str__(self) -> str:
        return str(self.atom)

    def __repr__(self) -> str:
        return f"Action({self})"


def static_predicates(schemas: Iterable[ActionSchema], dynamic: Iterable[str] = ()) -> Set[str]:
    """Predicates no action changes (types and topology such as Host(h), Link(u, v)).

    `dynamic` names predicates no action changes but the world can (facts the
    executor learns while running, e.g. Persistent(h)); they are never static.
    """
    schemas = list(schemas)
    changed = {a.pred for s in schemas for a in s.add + s.delete}
    used = {a.pred for s in schemas for a in s.pre_pos + s.pre_neg}
    return used - changed - set(dynamic)


@dataclass
class Problem:
    init: State
    goal: FrozenSet[Atom]
    actions: Tuple[Action, ...]
    negatable: FrozenSet[Atom] = frozenset()  # atoms p that have a compiled Not<p>
    urgency: float = 0.0

    def step_cost(self, a: Action) -> float:
        return (1.0 - self.urgency) * a.cost + self.urgency * a.duration

    def is_goal(self, state: State) -> bool:
        return self.goal <= state

    def complete(self, positive: Iterable[Atom]) -> State:
        """Add Not<p> for every negatable p that is absent (closed-world assumption)."""
        s = frozenset(positive) - self._not_atoms
        return s | frozenset(negated(p) for p in self.negatable if p not in s)

    @property
    def _not_atoms(self) -> FrozenSet[Atom]:
        return frozenset(negated(p) for p in self.negatable)

    def positive_view(self, state: State) -> State:
        """The state without compiled Not<p> atoms, for display."""
        nots = self._not_atoms
        return frozenset(a for a in state if a not in nots)

    def _index(self) -> Tuple[Dict[Atom, List[Action]], List[Action]]:
        """Actions by precondition atom, and actions with no preconditions (cached)."""
        cached = self.__dict__.get("_index_cache")
        if cached is None or cached[0] is not self.actions:
            consumers: Dict[Atom, List[Action]] = {}
            for a in self.actions:
                for p in a.pre:
                    consumers.setdefault(p, []).append(a)
            cached = (self.actions, consumers, [a for a in self.actions if not a.pre])
            self.__dict__["_index_cache"] = cached
        return cached[1], cached[2]

    def with_init(self, positive: Iterable[Atom]) -> "Problem":
        return Problem(self.complete(positive), self.goal, self.actions, self.negatable, self.urgency)

    def with_urgency(self, urgency: float) -> "Problem":
        return Problem(self.init, self.goal, self.actions, self.negatable, urgency)


def ground(
    schemas: Sequence[ActionSchema],
    objects: Sequence[str],
    init: Iterable[Atom],
    goal: Union[str, Iterable[Atom]],
    urgency: float = 0.0,
    cost_fn: Optional[Callable[[ActionSchema, Tuple[Const, ...]], Tuple[float, float]]] = None,
    dynamic: Iterable[str] = (),
) -> Problem:
    """Ground every schema over the objects and compile negative preconditions and goals.

    A grounding is kept only if its static preconditions hold in the initial state,
    which is what restricts e.g. Isolate(h) to hosts and BlockEdge(u, v) to real links.
    cost_fn(schema, args) may return a per-grounding (cost, duration).
    """
    init = frozenset(init)
    if isinstance(goal, str):
        goal_pos, goal_neg = literals(goal)
    else:
        goal_pos, goal_neg = list(goal), []
    static = static_predicates(schemas, dynamic)
    consts = [Const(o) for o in objects]

    raw: List[Tuple[ActionSchema, Tuple[Const, ...], List[Atom], List[Atom], List[Atom], List[Atom]]] = []
    for schema in schemas:
        for args in itertools.product(consts, repeat=len(schema.params)):
            theta = dict(zip(schema.params, args))
            pos = [substitute(a, theta) for a in schema.pre_pos]
            neg = [substitute(a, theta) for a in schema.pre_neg]
            if any(a.pred in static and a not in init for a in pos):
                continue
            if any(a.pred in static and a in init for a in neg):
                continue
            add = [substitute(a, theta) for a in schema.add]
            delete = [substitute(a, theta) for a in schema.delete]
            raw.append((schema, args, pos, neg, add, delete))

    negatable = frozenset({a for *_, neg, _, _ in raw for a in neg if a.pred not in static} | set(goal_neg))
    actions: List[Action] = []
    for schema, args, pos, neg, add, delete in raw:
        pre = {a for a in pos if a.pred not in static} | {negated(a) for a in neg if a.pred not in static}
        add_set, del_set = set(add), set(delete)
        for p in negatable:
            if p in add_set:
                del_set.add(negated(p))
            if p in del_set:
                add_set.add(negated(p))
        cost, duration = cost_fn(schema, args) if cost_fn else (schema.cost, schema.duration)
        actions.append(Action(schema.name, args, frozenset(pre), frozenset(add_set),
                              frozenset(del_set - add_set), cost, duration))

    problem = Problem(frozenset(), frozenset(goal_pos) | {negated(a) for a in goal_neg}, tuple(actions),
                      negatable, urgency)
    dynamic_init = {a for a in init if a.pred not in static}
    return problem.with_init(dynamic_init)


def urgency_from_ttc(ttc_hours: Optional[float], scale_hours: float = 24.0) -> float:
    """0 when there is plenty of time, approaching 1 as time-to-compromise approaches 0."""
    if ttc_hours is None:
        return 0.0
    return scale_hours / (scale_hours + max(0.0, ttc_hours))


# --------------------------------------------------------------------------- heuristics


def relaxed_costs(problem: Problem, state: State, combine: Callable[[Iterable[float]], float]) -> Dict[Atom, float]:
    """Cost of reaching each atom from `state` when delete effects are ignored.

    Generalized Dijkstra: atoms are finalized in order of cost; an action fires
    once all its preconditions are final, with cost combine(pre costs) + its own
    cost. combine is max (h_max) or sum (h_add); both are monotone, so popping
    atoms in cost order gives the same fixpoint as iterating over all actions.
    """
    consumers, no_pre = problem._index()
    cost: Dict[Atom, float] = {}
    heap: List[Tuple[float, int, Atom]] = []
    tie = itertools.count()
    for p in state:
        cost[p] = 0.0
        heap.append((0.0, next(tie), p))
    remaining = {a: len(a.pre) for a in problem.actions}
    acc: Dict[Action, float] = {}
    is_max = combine is _max

    def fire(a: Action, base: float) -> None:
        c = base + problem.step_cost(a)
        for q in a.add:
            if c < cost.get(q, INF):
                cost[q] = c
                heappush(heap, (c, next(tie), q))

    for a in no_pre:
        fire(a, 0.0)
    done: Set[Atom] = set()
    while heap:
        c, _, p = heappop(heap)
        if p in done or c > cost[p]:
            continue
        done.add(p)
        for a in consumers.get(p, ()):
            acc[a] = max(acc.get(a, 0.0), c) if is_max else acc.get(a, 0.0) + c
            remaining[a] -= 1
            if remaining[a] == 0:
                fire(a, acc[a])
    return cost


def _max(values: Iterable[float]) -> float:
    return max(values, default=0.0)


def _sum(values: Iterable[float]) -> float:
    return sum(values)


COMBINE = {"hmax": _max, "hadd": _sum}


def h_max(problem: Problem, state: State) -> float:
    cost = relaxed_costs(problem, state, _max)
    return _max(cost.get(g, INF) for g in problem.goal)


def h_add(problem: Problem, state: State) -> float:
    cost = relaxed_costs(problem, state, _sum)
    return _sum(cost.get(g, INF) for g in problem.goal)


# --------------------------------------------------------------------------- search


class PlanResult(NamedTuple):
    plan: Optional[List[Action]]
    cost: float
    nodes_expanded: int

    @property
    def solved(self) -> bool:
        return self.plan is not None


def progression(problem: Problem, heuristic: str = "hadd", max_nodes: int = 200_000) -> PlanResult:
    """A* forward search from the initial state. heuristic: "none", "hmax" or "hadd"."""
    if heuristic == "none":
        h = lambda s: 0.0  # noqa: E731
    elif heuristic == "hmax":
        h = lambda s: h_max(problem, s)  # noqa: E731
    elif heuristic == "hadd":
        h = lambda s: h_add(problem, s)  # noqa: E731
    else:
        raise ValueError(f"Unknown heuristic {heuristic!r}")

    counter = itertools.count()
    start = problem.init
    g_score: Dict[State, float] = {start: 0.0}
    parent: Dict[State, Tuple[Optional[State], Optional[Action]]] = {start: (None, None)}
    h0 = h(start)
    if h0 == INF:
        return PlanResult(None, INF, 0)
    frontier = [(h0, next(counter), start)]
    closed: Set[State] = set()
    expanded = 0
    while frontier:
        _, _, state = heappop(frontier)
        if state in closed:
            continue
        if problem.is_goal(state):
            return PlanResult(_extract(parent, state), g_score[state], expanded)
        closed.add(state)
        expanded += 1
        if expanded > max_nodes:
            break
        for a in problem.actions:
            if not a.applicable(state):
                continue
            nxt = a.apply(state)
            if nxt in closed:
                continue
            g = g_score[state] + problem.step_cost(a)
            if g < g_score.get(nxt, INF):
                hv = h(nxt)
                if hv == INF:
                    continue
                g_score[nxt] = g
                parent[nxt] = (state, a)
                heappush(frontier, (g + hv, next(counter), nxt))
    return PlanResult(None, INF, expanded)


def _extract(parent, state) -> List[Action]:
    plan: List[Action] = []
    while parent[state][0] is not None:
        state, action = parent[state]
        plan.append(action)
    plan.reverse()
    return plan


def regression(problem: Problem, heuristic: str = "hadd", max_nodes: int = 200_000) -> PlanResult:
    """A* backward search over goal sets.

    An action is relevant to goal set g if it adds some atom of g and deletes none.
    Sets that contain both p and Not<p> are pruned as inconsistent. The heuristic
    is computed once from the initial state: h(g) = combine of the relaxed cost of each atom.
    """
    if heuristic == "none":
        h = lambda g: 0.0  # noqa: E731
    else:
        combine = COMBINE[heuristic]
        atom_cost = relaxed_costs(problem, problem.init, combine)
        h = lambda g: combine(atom_cost.get(p, INF) for p in g)  # noqa: E731

    nots = {negated(p): p for p in problem.negatable}

    def consistent(g: FrozenSet[Atom]) -> bool:
        return not any(n in g and p in g for n, p in nots.items())

    counter = itertools.count()
    start = problem.goal
    g_score: Dict[FrozenSet[Atom], float] = {start: 0.0}
    parent: Dict[FrozenSet[Atom], Tuple[Optional[FrozenSet[Atom]], Optional[Action]]] = {start: (None, None)}
    frontier = [(h(start), next(counter), start)]
    closed: Set[FrozenSet[Atom]] = set()
    expanded = 0
    while frontier:
        _, _, g = heappop(frontier)
        if g in closed:
            continue
        if g <= problem.init:
            plan = _extract(parent, g)
            plan.reverse()  # actions were collected from the goal backwards
            return PlanResult(plan, g_score[g], expanded)
        closed.add(g)
        expanded += 1
        if expanded > max_nodes:
            break
        for a in problem.actions:
            if not (a.add & g) or (a.delete & g):
                continue
            g2 = (g - a.add) | a.pre
            if g2 in closed or not consistent(g2):
                continue
            cost = g_score[g] + problem.step_cost(a)
            if cost < g_score.get(g2, INF):
                hv = h(g2)
                if hv == INF:
                    continue
                g_score[g2] = cost
                parent[g2] = (g, a)
                heappush(frontier, (cost + hv, next(counter), g2))
    return PlanResult(None, INF, expanded)


def validate(problem: Problem, plan: Sequence[Action], state: Optional[State] = None) -> bool:
    """True if the plan is executable from `state` (default: init) and reaches the goal."""
    s = problem.init if state is None else state
    for a in plan:
        if not a.applicable(s):
            return False
        s = a.apply(s)
    return problem.is_goal(s)


def plan_cost(problem: Problem, plan: Sequence[Action]) -> float:
    return sum(problem.step_cost(a) for a in plan)


def compare_planners(problem: Problem, max_nodes: int = 200_000) -> List[Dict[str, object]]:
    """Nodes expanded and plan cost for both directions and all three heuristics."""
    rows = []
    for search in (progression, regression):
        for heuristic in ("none", "hmax", "hadd"):
            r = search(problem, heuristic, max_nodes)
            rows.append({"search": search.__name__, "heuristic": heuristic, "cost": r.cost,
                         "length": len(r.plan) if r.plan else None, "nodes_expanded": r.nodes_expanded})
    return rows


class STRIPSPlanner:
    """Convenience wrapper: plan(problem) with a chosen search direction and heuristic.

    Defaults to regression with h_max: optimal (h_max is admissible) and the fastest
    combination on the incident-response goals (see compare_planners).
    """

    def __init__(self, search: str = "regression", heuristic: str = "hmax"):
        self.search = {"progression": progression, "regression": regression}[search]
        self.heuristic = heuristic

    def plan(self, problem: Problem) -> Optional[List[Action]]:
        return self.search(problem, self.heuristic).plan


@dataclass
class Domain:
    """A set of schemas plus how to build problems from a scenario."""

    schemas: List[ActionSchema] = field(default_factory=list)

    def problem(self, objects: Sequence[str], init: Iterable[Atom], goal, urgency: float = 0.0,
                cost_fn=None, dynamic: Iterable[str] = ()) -> Problem:
        return ground(self.schemas, objects, init, goal, urgency, cost_fn, dynamic)


__all__ = [
    "Action", "ActionSchema", "Domain", "PlanResult", "Problem", "STRIPSPlanner", "atoms", "compare_planners",
    "ground", "h_add", "h_max", "literals", "negated", "plan_cost", "progression", "regression", "relaxed_costs",
    "urgency_from_ttc", "validate",
]

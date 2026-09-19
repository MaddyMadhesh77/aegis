"""Local search over firewall configurations.

A state is the frozenset of edges the firewall allows (a subset of the topology's
edges). The cost to minimize is

    sum over targets t of value(t) * P(t reachable from the entry)
    + flow_penalty * (number of required business flows the configuration breaks)

P(t reachable) is approximated by the most reliable path: the product of
per-edge exploit success probabilities along the best path, found with
Dijkstra on -log p. That is a lower bound on the true reachability probability,
which is #P-hard to compute exactly, and it is cheap enough to call thousands
of times. Edge success probability is max CVSS of the destination / 10, or
NO_CVE_SUCCESS when the destination has no known CVE.

A neighbour toggles one edge. hill_climb and simulated_annealing both record the
cost of the current state at every iteration so the report can plot them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from .graph_builder import critical_assets, max_cvss

Edge = Tuple[str, str]
State = FrozenSet[Edge]
NO_CVE_SUCCESS = 0.05


@dataclass
class LocalSearchResult:
    state: State
    cost: float
    history: List[float]          # cost of the current state after each iteration
    best_history: List[float]     # best cost seen so far after each iteration
    evaluations: int
    restarts: int = 0


@dataclass
class FirewallProblem:
    graph: nx.DiGraph
    entry: str = "Internet"
    targets: Optional[Sequence[str]] = None
    required_flows: Sequence[Edge] = ()
    flow_penalty: float = 5.0
    evaluations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.targets is None:
            self.targets = critical_assets(self.graph)
        self.edges: List[Edge] = sorted(self.graph.edges())
        self._neg_log_p: Dict[Edge, float] = {}
        for u, v in self.edges:
            attrs = self.graph.nodes[v]
            p = max_cvss(attrs) / 10.0 if attrs.get("cves") else NO_CVE_SUCCESS
            self._neg_log_p[(u, v)] = -math.log(max(p, 1e-9))

    # ------------------------------------------------------------------ states

    def all_open(self) -> State:
        return frozenset(self.edges)

    def random_state(self, rng: random.Random, p_open: float = 0.5) -> State:
        return frozenset(e for e in self.edges if rng.random() < p_open)

    def neighbors(self, state: State) -> Iterable[State]:
        for e in self.edges:
            yield state - {e} if e in state else state | {e}

    def random_neighbor(self, state: State, rng: random.Random) -> State:
        e = rng.choice(self.edges)
        return state - {e} if e in state else state | {e}

    # ------------------------------------------------------------------ objective

    def _subgraph(self, state: State) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nodes())
        g.add_edges_from((u, v, {"w": self._neg_log_p[(u, v)]}) for u, v in state)
        return g

    def reach_probabilities(self, state: State) -> Dict[str, float]:
        g = self._subgraph(state)
        dist = nx.single_source_dijkstra_path_length(g, self.entry, weight="w")
        return {t: math.exp(-dist[t]) if t in dist else 0.0 for t in self.targets}

    def broken_flows(self, state: State) -> List[Edge]:
        g = self._subgraph(state)
        return [(s, d) for s, d in self.required_flows if not nx.has_path(g, s, d)]

    def cost(self, state: State) -> float:
        self.evaluations += 1
        risk = sum(self.graph.nodes[t].get("value", 0.0) * p for t, p in self.reach_probabilities(state).items())
        return risk + self.flow_penalty * len(self.broken_flows(state))


def hill_climb(
    problem: FirewallProblem,
    restarts: int = 5,
    max_steps: int = 200,
    seed: Optional[int] = 0,
    start: Optional[State] = None,
) -> LocalSearchResult:
    """Steepest-descent hill climbing with random restarts.

    The first run starts from `start` (default: every edge open); each restart
    begins from a random configuration. History is concatenated across restarts.
    """
    rng = random.Random(seed)
    problem.evaluations = 0
    best_state, best_cost = None, math.inf
    history: List[float] = []
    best_history: List[float] = []
    for run in range(restarts + 1):
        state = (start if start is not None else problem.all_open()) if run == 0 else problem.random_state(rng)
        cost = problem.cost(state)
        for _ in range(max_steps):
            neighbor, n_cost = min(((n, problem.cost(n)) for n in problem.neighbors(state)), key=lambda x: x[1])
            if n_cost >= cost:
                break  # local minimum
            state, cost = neighbor, n_cost
            if cost < best_cost:
                best_state, best_cost = state, cost
            history.append(cost)
            best_history.append(best_cost)
        if cost < best_cost:
            best_state, best_cost = state, cost
        history.append(cost)
        best_history.append(best_cost)
    return LocalSearchResult(best_state, best_cost, history, best_history, problem.evaluations, restarts)


def simulated_annealing(
    problem: FirewallProblem,
    T0: float = 5.0,
    alpha: float = 0.995,
    steps: int = 3000,
    T_min: float = 1e-3,
    seed: Optional[int] = 0,
    start: Optional[State] = None,
) -> LocalSearchResult:
    """Simulated annealing with geometric cooling T_k = T0 * alpha^k.

    A worse neighbour is accepted with probability exp(-delta / T).
    """
    rng = random.Random(seed)
    problem.evaluations = 0
    state = start if start is not None else problem.all_open()
    cost = problem.cost(state)
    best_state, best_cost = state, cost
    history: List[float] = []
    best_history: List[float] = []
    T = T0
    for _ in range(steps):
        if T < T_min:
            break
        candidate = problem.random_neighbor(state, rng)
        c_cost = problem.cost(candidate)
        delta = c_cost - cost
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            state, cost = candidate, c_cost
            if cost < best_cost:
                best_state, best_cost = state, cost
        history.append(cost)
        best_history.append(best_cost)
        T *= alpha
    return LocalSearchResult(best_state, best_cost, history, best_history, problem.evaluations)

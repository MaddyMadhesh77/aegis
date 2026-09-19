"""Uninformed and informed path search over the attack graph.

Every search returns SearchResult(path, cost, nodes_expanded) so the report can
compare them on the same topology. `goal` may be one node or a collection of
nodes (any critical asset). A node counts as expanded when its successors are
generated. A failed search returns an empty path and infinite cost.

The A* heuristic is h(n) = w_min * hops(n, goal): w_min is the cheapest edge
weight and hops comes from one reverse BFS from the goals. Each edge lowers the
hop count by at most 1 and costs at least w_min, so h(n) <= c(n, n') + h(n') on
every edge. h is consistent, and therefore admissible.
"""

from __future__ import annotations

import itertools
from collections import deque
from heapq import heappop, heappush
from typing import Callable, Collection, Dict, Hashable, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple, Union

import networkx as nx

from .graph_builder import from_adjacency, max_cvss

Node = Hashable
Goal = Union[Node, Collection[Node]]
Heuristic = Callable[[Node], float]
INF = float("inf")


class SearchResult(NamedTuple):
    path: List[Node]
    cost: float
    nodes_expanded: int


def as_graph(graph: Union[nx.DiGraph, Mapping[Node, Mapping[Node, float]]]) -> nx.DiGraph:
    return graph if isinstance(graph, nx.DiGraph) else from_adjacency(graph)


def _goal_set(goal: Goal) -> Set[Node]:
    if isinstance(goal, (str, tuple)) or not isinstance(goal, Iterable):
        return {goal}
    return set(goal)


def reconstruct(came_from: Mapping[Node, Optional[Node]], goal: Node) -> List[Node]:
    """Walk parent pointers back from goal to the start (the node whose parent is None)."""
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def path_cost(graph: nx.DiGraph, path: List[Node]) -> float:
    if not path:
        return INF
    return sum(graph[u][v].get("weight", 1.0) for u, v in zip(path, path[1:]))


def _result(graph: nx.DiGraph, path: List[Node], expanded: int) -> SearchResult:
    return SearchResult(path, path_cost(graph, path), expanded)


# --------------------------------------------------------------------------- uninformed


def bfs(graph, start: Node, goal: Goal) -> SearchResult:
    """Fewest hops (not lowest cost). Goal test when a node is generated."""
    g, goals = as_graph(graph), _goal_set(goal)
    if start in goals:
        return _result(g, [start], 0)
    came_from: Dict[Node, Optional[Node]] = {start: None}
    frontier = deque([start])
    expanded = 0
    while frontier:
        node = frontier.popleft()
        expanded += 1
        for nbr in g.successors(node):
            if nbr not in came_from:
                came_from[nbr] = node
                if nbr in goals:
                    return _result(g, reconstruct(came_from, nbr), expanded)
                frontier.append(nbr)
    return SearchResult([], INF, expanded)


def dfs(graph, start: Node, goal: Goal) -> SearchResult:
    """Iterative depth-first graph search with an explicit stack."""
    g, goals = as_graph(graph), _goal_set(goal)
    came_from: Dict[Node, Optional[Node]] = {start: None}
    stack = [start]
    explored: Set[Node] = set()
    expanded = 0
    while stack:
        node = stack.pop()
        if node in goals:
            return _result(g, reconstruct(came_from, node), expanded)
        if node in explored:
            continue
        explored.add(node)
        expanded += 1
        for nbr in reversed(list(g.successors(node))):  # reversed: visit neighbours in graph order
            if nbr not in explored:
                came_from[nbr] = node
                stack.append(nbr)
    return SearchResult([], INF, expanded)


def depth_limited(graph, start: Node, goal: Goal, limit: int) -> SearchResult:
    """DFS to depth `limit`, with a cycle check along the current path only."""
    g, goals = as_graph(graph), _goal_set(goal)
    expanded = 0

    def recurse(path: List[Node], depth: int) -> Optional[List[Node]]:
        nonlocal expanded
        node = path[-1]
        if node in goals:
            return path
        if depth == 0:
            return None
        expanded += 1
        for nbr in g.successors(node):
            if nbr not in path:
                found = recurse(path + [nbr], depth - 1)
                if found is not None:
                    return found
        return None

    found = recurse([start], limit)
    return _result(g, found, expanded) if found else SearchResult([], INF, expanded)


def iddfs(graph, start: Node, goal: Goal, max_depth: int = 50) -> SearchResult:
    """Iterative deepening: depth-limited DFS with limits 0, 1, ..., max_depth.

    nodes_expanded counts the expansions of every iteration.
    """
    g = as_graph(graph)
    total = 0
    for limit in range(max_depth + 1):
        result = depth_limited(g, start, goal, limit)
        total += result.nodes_expanded
        if result.path:
            return SearchResult(result.path, result.cost, total)
    return SearchResult([], INF, total)


def ucs(graph, start: Node, goal: Goal) -> SearchResult:
    """Uniform-cost search: A* with h = 0."""
    return astar(graph, start, goal, h=lambda n: 0.0)


# --------------------------------------------------------------------------- informed


def hop_distances(graph, goal: Goal) -> Dict[Node, int]:
    """Fewest hops from each node to the nearest goal (reverse BFS from the goals)."""
    g, goals = as_graph(graph), _goal_set(goal)
    dist = {t: 0 for t in goals if t in g}
    frontier = deque(dist)
    while frontier:
        node = frontier.popleft()
        for pred in g.predecessors(node):
            if pred not in dist:
                dist[pred] = dist[node] + 1
                frontier.append(pred)
    return dist


def min_edge_weight(graph) -> float:
    g = as_graph(graph)
    return min((d.get("weight", 1.0) for _, _, d in g.edges(data=True)), default=0.0)


def hop_heuristic(graph, goal: Goal) -> Heuristic:
    """h(n) = w_min * hops(n, goal); infinite where no goal is reachable."""
    hops = hop_distances(graph, goal)
    w_min = min_edge_weight(graph)
    return lambda n: w_min * hops[n] if n in hops else INF


def cvss_hop_heuristic(graph, goal: Goal) -> Heuristic:
    """The heuristic from the original design, CVSS(n) * hops(n, goal).

    Kept only to show in the report that it is NOT admissible (verify_consistency fails).
    """
    g = as_graph(graph)
    hops = hop_distances(g, goal)
    return lambda n: max_cvss(g.nodes[n]) * hops[n] if n in hops else INF


def astar(graph, start: Node, goal: Goal, h: Optional[Heuristic] = None) -> SearchResult:
    """A* graph search with a closed set. Optimal when h is consistent.

    Ties on f are broken towards larger g (deeper nodes), then insertion order.
    """
    g, goals = as_graph(graph), _goal_set(goal)
    if h is None:
        h = hop_heuristic(g, goals)
    counter = itertools.count()
    g_score: Dict[Node, float] = {start: 0.0}
    came_from: Dict[Node, Optional[Node]] = {start: None}
    frontier: List[Tuple[float, float, int, Node]] = [(h(start), 0.0, next(counter), start)]
    closed: Set[Node] = set()
    expanded = 0
    while frontier:
        _, neg_g, _, node = heappop(frontier)
        if node in closed:
            continue  # stale queue entry
        if node in goals:
            return SearchResult(reconstruct(came_from, node), g_score[node], expanded)
        closed.add(node)
        expanded += 1
        for nbr in g.successors(node):
            if nbr in closed:
                continue
            tentative = g_score[node] + g[node][nbr].get("weight", 1.0)
            if tentative < g_score.get(nbr, INF):
                g_score[nbr] = tentative
                came_from[nbr] = node
                f = tentative + h(nbr)
                if f < INF:
                    heappush(frontier, (f, -tentative, next(counter), nbr))
    return SearchResult([], INF, expanded)


def verify_consistency(graph, h: Heuristic, goal: Goal, tol: float = 1e-9) -> bool:
    """Assert h(goal) = 0 and h(n) <= c(n, n') + h(n') on every edge. Returns True."""
    g, goals = as_graph(graph), _goal_set(goal)
    for t in goals:
        assert h(t) == 0, f"h({t}) = {h(t)}, expected 0 at a goal"
    for u, v, data in g.edges(data=True):
        hu, hv, c = h(u), h(v), data.get("weight", 1.0)
        if hu == INF:
            continue  # no goal reachable from u: nothing to bound
        assert hu <= c + hv + tol, f"inconsistent on {u} -> {v}: h={hu} > c={c} + h'={hv}"
    return True


# --------------------------------------------------------------------------- comparison


ALGORITHMS: Dict[str, Callable[..., SearchResult]] = {
    "bfs": bfs, "dfs": dfs, "iddfs": iddfs, "ucs": ucs, "astar": astar,
}


def compare_algorithms(graph, start: Node, goal: Goal) -> List[Dict[str, object]]:
    """One row per algorithm, plus networkx Dijkstra as the reference optimal cost."""
    g = as_graph(graph)
    rows: List[Dict[str, object]] = []
    for name, fn in ALGORITHMS.items():
        r = fn(g, start, goal)
        rows.append({"algorithm": name, "cost": r.cost, "hops": max(len(r.path) - 1, 0),
                     "nodes_expanded": r.nodes_expanded, "path": r.path})
    goals = _goal_set(goal)
    lengths = nx.single_source_dijkstra_path_length(g, start, weight="weight")
    rows.append({"algorithm": "networkx_dijkstra", "cost": min((lengths[t] for t in goals if t in lengths), default=INF),
                 "hops": None, "nodes_expanded": None, "path": None})
    return rows


def choke_edges(graph, start: Node, goal: Goal) -> Set[Tuple[Node, Node]]:
    """A minimum set of edges whose removal disconnects start from every goal.

    Every real edge has capacity 1; the edges into the added super-sink have no
    capacity attribute, which networkx treats as infinite, so they are never cut.
    """
    g = nx.DiGraph()
    g.add_edges_from(as_graph(graph).edges(), capacity=1)
    sink = ("__sink__",)
    for t in _goal_set(goal):
        g.add_edge(t, sink)
    _, (reachable, _) = nx.minimum_cut(g, start, sink)
    return {(u, v) for u, v in g.edges() if u in reachable and v not in reachable}


# Backwards-compatible names used by older code: they return only the path.
def bfs_search(graph, start: Node, target: Node) -> List[Node]:
    return bfs(graph, start, target).path


def dfs_search(graph, start: Node, target: Node) -> List[Node]:
    return dfs(graph, start, target).path


def astar_search(graph, start: Node, target: Node) -> List[Node]:
    return astar(graph, start, target).path

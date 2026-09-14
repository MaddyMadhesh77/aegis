from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def build_graph(nodes: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = {}
    for node, neighbors in nodes.items():
        graph[node] = dict(neighbors)
    for node in graph:
        if node not in graph:
            graph[node] = {}
    return graph

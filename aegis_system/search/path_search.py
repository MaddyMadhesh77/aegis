from __future__ import annotations

from heapq import heappop, heappush
from typing import Dict, List, Tuple


def bfs_search(graph: Dict[str, Dict[str, float]], start: str, target: str) -> List[str]:
    queue = [start]
    visited = {start}
    parent = {start: None}

    while queue:
        node = queue.pop(0)
        if node == target:
            break
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path if path and path[0] == start else []


def dfs_search(graph: Dict[str, Dict[str, float]], start: str, target: str) -> List[str]:
    stack = [(start, [start])]
    visited = {start}

    while stack:
        node, path = stack.pop()
        if node == target:
            return path
        for neighbor in reversed(list(graph.get(node, {}))):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    return []


def astar_search(graph: Dict[str, Dict[str, float]], start: str, target: str) -> List[str]:
    frontier: List[Tuple[float, int, str]] = []
    heappush(frontier, (0, 0, start))
    g_score = {start: 0.0}
    came_from = {start: None}

    while frontier:
        _, _, current = heappop(frontier)
        if current == target:
            break
        for neighbor, weight in graph.get(current, {}).items():
            tentative_g = g_score[current] + weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heuristic = 0.0
                heappush(frontier, (tentative_g + heuristic, len(g_score), neighbor))

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path if path and path[0] == start else []

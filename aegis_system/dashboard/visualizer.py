from __future__ import annotations

from typing import Any, Dict, List


def render_network_summary(graph: Dict[str, List[str]]) -> Dict[str, Any]:
    nodes = sorted(graph.keys())
    edges = [
        {"source": source, "target": target}
        for source, targets in graph.items()
        for target in targets
    ]
    return {"nodes": nodes, "edges": edges}

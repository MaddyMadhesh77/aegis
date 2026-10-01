"""Provenance graph linking every pipeline result to the results it came from.

The dashboard uses `lineage` to walk an action back through
plan -> proof -> belief -> detection.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class TraceNode:
    trace_id: str
    stage: str
    summary: str
    payload: Dict[str, Any] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    seq: int = 0


class TraceStore:
    def __init__(self) -> None:
        self._nodes: Dict[str, TraceNode] = {}
        self._counter = itertools.count(1)

    def record(
        self,
        stage: str,
        summary: str,
        payload: Optional[Dict[str, Any]] = None,
        parents: Iterable[Optional[str]] = (),
    ) -> str:
        parent_ids = [p for p in parents if p is not None]
        for p in parent_ids:
            if p not in self._nodes:
                raise KeyError(f"Unknown parent trace id {p!r}")
        seq = next(self._counter)
        trace_id = f"{stage}-{seq}"
        self._nodes[trace_id] = TraceNode(trace_id, stage, summary, dict(payload or {}), parent_ids, seq)
        return trace_id

    def get(self, trace_id: str) -> TraceNode:
        return self._nodes[trace_id]

    def __contains__(self, trace_id: str) -> bool:
        return trace_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def lineage(self, trace_id: str) -> List[TraceNode]:
        """The node and all its ancestors, nearest first (breadth-first)."""
        result: List[TraceNode] = []
        seen = {trace_id}
        queue = deque([trace_id])
        while queue:
            node = self._nodes[queue.popleft()]
            result.append(node)
            for parent in node.parents:
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return result

    def by_stage(self, stage: str) -> List[TraceNode]:
        return sorted((n for n in self._nodes.values() if n.stage == stage), key=lambda n: n.seq)

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> "TraceStore":
        """Rebuild a store saved with to_dict (parents always precede children in insertion order)."""
        store = cls()
        for seq, (tid, n) in enumerate(data.items(), start=1):
            store._nodes[tid] = TraceNode(tid, n["stage"], n["summary"], dict(n.get("payload", {})),
                                          list(n.get("parents", [])), seq)
        store._counter = itertools.count(len(data) + 1)
        return store

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            tid: {"stage": n.stage, "summary": n.summary, "payload": n.payload, "parents": list(n.parents)}
            for tid, n in self._nodes.items()
        }

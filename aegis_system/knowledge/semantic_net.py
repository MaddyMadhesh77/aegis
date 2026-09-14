from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Set, Tuple


class SemanticNet:
    def __init__(self):
        self.relations: DefaultDict[str, Dict[str, Set[str]]] = defaultdict(dict)

    def add_relation(self, source: str, relation: str, target: str) -> None:
        self.relations[source].setdefault(relation, set()).add(target)

    def has_relation(self, source: str, relation: str, target: str) -> bool:
        return target in self.relations.get(source, {}).get(relation, set())

    def path_exists(self, source: str, target: str) -> bool:
        queue = deque([source])
        seen = {source}

        while queue:
            current = queue.popleft()
            for neighbors in self.relations.get(current, {}).values():
                for neighbor in neighbors:
                    if neighbor == target:
                        return True
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
        return False

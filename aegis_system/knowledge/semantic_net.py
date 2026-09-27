"""Semantic network of ATT&CK techniques, tactics, mitigations and tools.

Nodes carry attributes; labelled edges relate them. Inheritance follows the
taxonomic relations upwards (instance_of, is_a, subtechnique_of): `inherits`
returns the value from the nearest node that defines the attribute, so
T1003.001 (LSASS Memory) inherits requires_privilege = "SYSTEM" from T1003.

Relations used by the ATT&CK loader
    subtechnique_of   T1003.001 -> T1003
    is_a              T1003 -> Technique, Mimikatz -> Tool
    instance_of       individual hosts / alerts -> their class
    part_of           T1003 -> credential-access (tactic)
    mitigated_by      T1003 -> M1043
    implements        Mimikatz -> T1003.001
    detected_as       alert class -> technique
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Union

ATTACK_JSON = Path(__file__).with_name("data") / "attack_subset.json"
INHERIT_RELATIONS = ("instance_of", "is_a", "subtechnique_of")


class SemanticNet:
    def __init__(self) -> None:
        self.relations: DefaultDict[str, Dict[str, Set[str]]] = defaultdict(dict)
        self.attributes: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)

    # ----------------------------------------------------------------- building

    def add_relation(self, source: str, relation: str, target: str) -> None:
        self.relations[source].setdefault(relation, set()).add(target)
        self.relations.setdefault(target, {})

    def set_attr(self, node: str, attr: str, value: Any) -> None:
        self.attributes[node][attr] = value
        self.relations.setdefault(node, {})

    # ----------------------------------------------------------------- queries

    @property
    def nodes(self) -> List[str]:
        return sorted(set(self.relations) | set(self.attributes))

    def has_relation(self, source: str, relation: str, target: str) -> bool:
        return target in self.relations.get(source, {}).get(relation, set())

    def related(self, source: str, relation: str) -> Set[str]:
        return set(self.relations.get(source, {}).get(relation, set()))

    def sources(self, relation: str, target: str) -> Set[str]:
        """Nodes with an edge `relation` into target (the inverse direction)."""
        return {s for s, rels in self.relations.items() if target in rels.get(relation, set())}

    def path_exists(self, source: str, target: str) -> bool:
        """Is target reachable from source along edges of any relation?"""
        queue = deque([source])
        seen = {source}
        while queue:
            current = queue.popleft()
            for neighbors in self.relations.get(current, {}).values():
                for neighbor in sorted(neighbors):
                    if neighbor == target:
                        return True
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
        return False

    def ancestors(self, node: str, relations: Iterable[str] = INHERIT_RELATIONS) -> List[str]:
        """Taxonomic ancestors in breadth-first order (nearest first), excluding the node itself."""
        relations = tuple(relations)
        order: List[str] = []
        seen = {node}
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for rel in relations:
                for parent in sorted(self.related(current, rel)):
                    if parent not in seen:
                        seen.add(parent)
                        order.append(parent)
                        queue.append(parent)
        return order

    def inherits(self, node: str, attr: str, default: Any = None) -> Any:
        """The attribute's value on the node or on its nearest ancestor that defines it."""
        for n in [node] + self.ancestors(node):
            if attr in self.attributes.get(n, {}):
                return self.attributes[n][attr]
        return default

    def is_a(self, node: str, category: str) -> bool:
        return node == category or category in self.ancestors(node)

    def transitive_closure(self, relation: str) -> Set[Tuple[str, str]]:
        """Every pair (a, c) connected by a chain of one or more `relation` edges."""
        closure: Set[Tuple[str, str]] = set()
        for start in list(self.relations):
            stack = list(self.related(start, relation))
            seen: Set[str] = set()
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                closure.add((start, n))
                stack.extend(self.related(n, relation))
        return closure

    # ----------------------------------------------------------------- ATT&CK helpers

    def mitigations_for(self, technique: str) -> List[str]:
        """Mitigations of the technique and of every technique it specializes."""
        found: Set[str] = set()
        for t in [technique] + self.ancestors(technique, ("subtechnique_of",)):
            found |= self.related(t, "mitigated_by")
        return sorted(found)

    def tactic_of(self, technique: str) -> Optional[str]:
        for t in [technique] + self.ancestors(technique, ("subtechnique_of",)):
            tactics = self.related(t, "part_of")
            if tactics:
                return sorted(tactics)[0]
        return None

    def technique_for_alert(self, alert_class: str) -> Optional[str]:
        targets = self.related(alert_class, "detected_as")
        return sorted(targets)[0] if targets else None

    def techniques_by_tactic(self, tactic: str) -> List[str]:
        return sorted(self.sources("part_of", tactic))

    def name(self, node: str) -> str:
        return self.attributes.get(node, {}).get("name", node)

    # ----------------------------------------------------------------- loading

    @classmethod
    def from_attack_json(cls, path: Union[str, Path, None] = None) -> "SemanticNet":
        data = json.loads(Path(path or ATTACK_JSON).read_text())
        net = cls()
        for tid, t in data["tactics"].items():
            net.add_relation(tid, "is_a", "Tactic")
            net.set_attr(tid, "name", t["name"])
            net.set_attr(tid, "kill_chain_phase", t["kill_chain_phase"])
        for mid, name in data["mitigations"].items():
            net.add_relation(mid, "is_a", "Mitigation")
            net.set_attr(mid, "name", name)
        for tech_id, t in data["techniques"].items():
            if "parent" in t:
                net.add_relation(tech_id, "subtechnique_of", t["parent"])
            else:
                net.add_relation(tech_id, "is_a", "Technique")
            net.add_relation(tech_id, "part_of", t["tactic"])
            for key, value in t.items():
                if key not in ("parent", "tactic", "mitigations"):
                    net.set_attr(tech_id, key, value)
            for m in t.get("mitigations", []):
                net.add_relation(tech_id, "mitigated_by", m)
        net.set_attr("Technique", "requires_privilege", "user")  # default, overridden per technique
        for tool, info in data.get("tools", {}).items():
            net.add_relation(tool, "is_a", "Tool")
            for tech in info.get("implements", []):
                net.add_relation(tool, "implements", tech)
        for cls_name, tech in data.get("alert_classes", {}).items():
            if not cls_name.startswith("_"):
                net.add_relation(cls_name, "detected_as", tech)
        return net

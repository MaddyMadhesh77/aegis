"""OWL ontology of the network and the ATT&CK subset, built with owlready2.

Classes      Host (Workstation, Server, DomainController, Database, DMZHost),
             Vulnerability, CriticalVulnerability, Technique (one subclass per
             tactic), Mitigation, ExposedHost
Properties   exploits (Technique -> Vulnerability), hasVulnerability (Host ->
             Vulnerability), lateralMoveTo (Host -> Host, transitive),
             mitigatedBy (Technique -> Mitigation), subtechniqueOf, partOf (Technique -> Tactic),
             cvssScore (Vulnerability -> float)

Defined classes, filled in by the reasoner:
    CriticalVulnerability == Vulnerability and cvssScore some float[>= 9.0]
    ExposedHost           == Host and hasVulnerability some CriticalVulnerability

`reason()` runs HermiT, which needs Java. When Java is missing (or HermiT fails)
it falls back to `_fallback_reason`, which computes the same defined classes and
the transitive closure of lateralMoveTo in Python, so SPARQL queries keep working.
Each manager uses its own owlready2 World, so tests and scenarios do not share state.
Classes and properties use the prefix a:, individuals the prefix d:.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import networkx as nx

try:
    import owlready2 as owl
except ImportError:  # pragma: no cover - exercised only on machines without owlready2
    owl = None

from .semantic_net import ATTACK_JSON

IRI = "http://aegis.example/ontology#"
DATA_IRI = "http://aegis.example/data#"  # individuals live apart from classes (a host may be called DomainController)
CRITICAL_CVSS = 9.0
HOST_TYPES = {"workstation": "Workstation", "server": "Server", "domain_controller": "DomainController",
              "database": "Database", "dmz": "DMZHost"}


def _local(name: str) -> str:
    """OWL individual names: ATT&CK ids contain '.', which is fine in IRIs but awkward in SPARQL; use '_'."""
    return name.replace(".", "_").replace("-", "_")


class OntologyManager:
    def __init__(self, attack_json: Union[str, Path, None] = None):
        if owl is None:
            raise ImportError("owlready2 is required for the ontology layer (pip install owlready2)")
        self.world = owl.World()
        self.onto = self.world.get_ontology(IRI.rstrip("#"))
        self.data = self.onto.get_namespace(DATA_IRI)
        self.reasoner_used: Optional[str] = None
        self._build_schema()
        self._load_attack(json.loads(Path(attack_json or ATTACK_JSON).read_text()))

    # ----------------------------------------------------------------- schema

    def _build_schema(self) -> None:
        with self.onto:
            class Host(owl.Thing): pass
            class Vulnerability(owl.Thing): pass
            class Technique(owl.Thing): pass
            class Tactic(owl.Thing): pass
            class Mitigation(owl.Thing): pass
            for cls in HOST_TYPES.values():
                type(cls, (Host,), {})

            class exploits(owl.ObjectProperty):
                domain = [Technique]
                range = [Vulnerability]

            class hasVulnerability(owl.ObjectProperty):
                domain = [Host]
                range = [Vulnerability]

            class lateralMoveTo(owl.ObjectProperty, owl.TransitiveProperty):
                domain = [Host]
                range = [Host]

            class mitigatedBy(owl.ObjectProperty):
                domain = [Technique]
                range = [Mitigation]

            class subtechniqueOf(owl.ObjectProperty):
                domain = [Technique]
                range = [Technique]

            class partOf(owl.ObjectProperty):
                domain = [Technique]
                range = [Tactic]

            class cvssScore(owl.DataProperty, owl.FunctionalProperty):
                domain = [Vulnerability]
                range = [float]

            class CriticalVulnerability(Vulnerability):
                equivalent_to = [Vulnerability & cvssScore.some(owl.ConstrainedDatatype(float, min_inclusive=CRITICAL_CVSS))]

            class ExposedHost(Host):
                equivalent_to = [Host & hasVulnerability.some(CriticalVulnerability)]

    def cls(self, name: str):
        return getattr(self.onto, name)

    def individual(self, name: str):
        return self.world[DATA_IRI + _local(name)]

    def _new(self, cls_name_or_cls, name: str):
        cls = self.cls(cls_name_or_cls) if isinstance(cls_name_or_cls, str) else cls_name_or_cls
        return cls(_local(name), namespace=self.data)

    # ----------------------------------------------------------------- data

    def _load_attack(self, data: Dict) -> None:
        with self.onto:
            tactic_classes = {}
            for tid, t in data["tactics"].items():
                cls_name = t["name"].replace(" ", "") + "Technique"
                tactic_classes[tid] = type(cls_name, (self.cls("Technique"),), {})
                self._new("Tactic", tid)
            for mid, name in data["mitigations"].items():
                m = self._new("Mitigation", mid)
                m.label = [name]
            for tech_id, t in data["techniques"].items():
                tech = self._new(tactic_classes[t["tactic"]], tech_id)
                tech.label = [t["name"]]
                tech.partOf = [self.individual(t["tactic"])]
                tech.mitigatedBy = [self.individual(m) for m in t.get("mitigations", [])]
            for tech_id, t in data["techniques"].items():
                if "parent" in t:
                    self.individual(tech_id).subtechniqueOf = [self.individual(t["parent"])]

    def load_topology(self, graph: nx.DiGraph, exploited_by: Optional[Dict[str, Sequence[str]]] = None) -> None:
        """Hosts, their vulnerabilities, and lateralMoveTo for every network link between hosts.

        exploited_by maps a CVE id to technique ids that exploit it (optional).
        """
        exploited_by = exploited_by or {}
        with self.onto:
            for name, attrs in graph.nodes(data=True):
                if attrs.get("type") == "internet":
                    continue
                host = self._new(HOST_TYPES.get(attrs.get("type", ""), "Host"), name)
                for cve in attrs.get("cves", []):
                    vuln = self._new("Vulnerability", cve["id"])
                    vuln.cvssScore = float(cve["cvss"])
                    host.hasVulnerability.append(vuln)
                    for tech in exploited_by.get(cve["id"], []):
                        self.individual(tech).exploits.append(vuln)
            for u, v in graph.edges():
                if graph.nodes[u].get("type") != "internet" and graph.nodes[v].get("type") != "internet":
                    self.individual(u).lateralMoveTo.append(self.individual(v))

    # ----------------------------------------------------------------- reasoning

    @staticmethod
    def java_available() -> bool:
        return shutil.which("java") is not None

    def reason(self, use_hermit: Optional[bool] = None) -> str:
        """Classify the ontology. Returns "hermit" or "fallback"."""
        if use_hermit is None:
            use_hermit = self.java_available()
        if use_hermit:
            try:
                owl.sync_reasoner_hermit(self.world, infer_property_values=True, debug=0)
                self.reasoner_used = "hermit"
                return self.reasoner_used
            except Exception:  # noqa: BLE001 - any reasoner failure falls back to the Python version
                pass
        self._fallback_reason()
        self.reasoner_used = "fallback"
        return self.reasoner_used

    def _fallback_reason(self) -> None:
        """Python version of what the reasoner infers for this ontology."""
        critical = self.cls("CriticalVulnerability")
        exposed = self.cls("ExposedHost")
        with self.onto:
            for v in self.cls("Vulnerability").instances():
                if v.cvssScore is not None and v.cvssScore >= CRITICAL_CVSS and critical not in v.is_a:
                    v.is_a.append(critical)
            for h in self.cls("Host").instances():
                if any(critical in v.is_a for v in h.hasVulnerability) and exposed not in h.is_a:
                    h.is_a.append(exposed)
            # transitive closure of lateralMoveTo, stored explicitly
            hosts = list(self.cls("Host").instances())
            reach = {h: set(h.lateralMoveTo) for h in hosts}
            changed = True
            while changed:
                changed = False
                for h in hosts:
                    extra = set().union(*(reach[n] for n in reach[h] if n in reach)) - reach[h]
                    if extra:
                        reach[h] |= extra
                        changed = True
            for h in hosts:
                for n in sorted(reach[h] - set(h.lateralMoveTo), key=lambda x: x.name):
                    h.lateralMoveTo.append(n)

    def is_subclass(self, sub: str, sup: str) -> bool:
        """Subsumption over the asserted (and, after reasoning, inferred) class hierarchy."""
        return self.cls(sup) in self.cls(sub).ancestors()

    # ----------------------------------------------------------------- queries

    def sparql(self, query: str) -> List[List]:
        prefix = (f"PREFIX a: <{IRI}>\nPREFIX d: <{DATA_IRI}>\n"
                  "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n")
        return [list(row) for row in self.world.sparql(prefix + query)]

    def names(self, rows: List[List]) -> List[str]:
        return sorted(r[0].name if hasattr(r[0], "name") else str(r[0]) for r in rows)

    def exposed_hosts(self) -> List[str]:
        return self.names(self.sparql("SELECT ?h WHERE { ?h rdf:type a:ExposedHost . }"))

    def critical_vulnerabilities(self) -> List[str]:
        return self.names(self.sparql("SELECT ?v WHERE { ?v rdf:type a:CriticalVulnerability . }"))

    def reachable_from(self, host: str) -> List[str]:
        """Hosts reachable by lateral movement (needs reason() for the transitive closure)."""
        return self.names(self.sparql(f"SELECT ?h WHERE {{ d:{_local(host)} a:lateralMoveTo ?h . }}"))

    def mitigations_for(self, technique: str) -> List[str]:
        """Mitigations of the technique and of the techniques it specializes (property path subtechniqueOf*)."""
        q = f"SELECT DISTINCT ?m WHERE {{ d:{_local(technique)} a:subtechniqueOf* ?t . ?t a:mitigatedBy ?m . }}"
        return self.names(self.sparql(q))

    def techniques_of_tactic(self, tactic: str) -> List[str]:
        return self.names(self.sparql(f"SELECT ?t WHERE {{ ?t a:partOf d:{_local(tactic)} . }}"))

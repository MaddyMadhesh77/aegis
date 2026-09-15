"""Network topology as an attributed directed graph G = (V, E).

Node attributes
    type             internet | dmz | workstation | server | domain_controller | database
    cves             list of {"id": str, "cvss": float}
    services         list of service names
    criticality      0..1, cost of taking the host offline
    value            asset value an attacker gains by compromising it
    admin_credentials  True if the host holds cached admin credentials

Edge (u, v) means a firewall rule lets u reach a service on v. Its weight is the
exploit difficulty of v:  w = max(MIN_WEIGHT, 10 - highest CVSS on v).
A host with no known CVE costs NO_CVE_WEIGHT (needs stolen credentials).
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

MIN_WEIGHT = 0.5
NO_CVE_WEIGHT = 10.0

NODE_DEFAULTS = {
    "type": "workstation",
    "cves": [],
    "services": [],
    "criticality": 0.1,
    "value": 0.0,
    "admin_credentials": False,
}


def max_cvss(attrs: Mapping) -> float:
    return max((c["cvss"] for c in attrs.get("cves", [])), default=0.0)


def exploit_difficulty(target_attrs: Mapping) -> float:
    if not target_attrs.get("cves"):
        return NO_CVE_WEIGHT
    return max(MIN_WEIGHT, 10.0 - max_cvss(target_attrs))


def build_graph(hosts: Mapping[str, Mapping], links: Iterable[Tuple[str, str]]) -> nx.DiGraph:
    """Build the topology from host attributes and allowed (source, destination) links."""
    graph = nx.DiGraph()
    for name, attrs in hosts.items():
        graph.add_node(name, **{**NODE_DEFAULTS, **attrs})
    for u, v in links:
        for node in (u, v):
            if node not in graph:
                graph.add_node(node, **NODE_DEFAULTS)
        graph.add_edge(u, v, weight=exploit_difficulty(graph.nodes[v]))
    return graph


def from_adjacency(adjacency: Mapping[str, Mapping[str, float]]) -> nx.DiGraph:
    """Build a plain weighted graph from {node: {neighbor: weight}}.

    Nodes that only appear as neighbors are added too.
    """
    graph = nx.DiGraph()
    for u, neighbors in adjacency.items():
        graph.add_node(u)
        for v, w in neighbors.items():
            graph.add_edge(u, v, weight=float(w))
    return graph


def example_topology() -> nx.DiGraph:
    """A small hand-written enterprise network used in tests and demos.

    CVE identifiers are simulated; scores are illustrative.
    """
    hosts = {
        "Internet": {"type": "internet", "criticality": 0.0},
        "WebServer": {
            "type": "dmz", "services": ["https"], "criticality": 0.6, "value": 2.0,
            "cves": [{"id": "CVE-SIM-0101", "cvss": 9.8}],
        },
        "MailServer": {
            "type": "dmz", "services": ["smtp"], "criticality": 0.5, "value": 2.0,
            "cves": [{"id": "CVE-SIM-0102", "cvss": 7.5}],
        },
        "Workstation1": {
            "type": "workstation", "services": ["smb"], "criticality": 0.1, "value": 1.0,
            "cves": [{"id": "CVE-SIM-0201", "cvss": 8.1}],
        },
        "Workstation2": {
            "type": "workstation", "services": ["rdp"], "criticality": 0.1, "value": 1.0,
            "cves": [{"id": "CVE-SIM-0202", "cvss": 6.5}], "admin_credentials": True,
        },
        "AppServer": {
            "type": "server", "services": ["http"], "criticality": 0.7, "value": 4.0,
            "cves": [{"id": "CVE-SIM-0301", "cvss": 8.8}],
        },
        "FileServer": {
            "type": "server", "services": ["smb"], "criticality": 0.6, "value": 5.0,
            "cves": [{"id": "CVE-SIM-0302", "cvss": 7.8}],
        },
        "DomainController": {
            "type": "domain_controller", "services": ["ldap", "kerberos"], "criticality": 1.0,
            "value": 10.0, "cves": [{"id": "CVE-SIM-0401", "cvss": 9.0}], "admin_credentials": True,
        },
        "Database": {
            "type": "database", "services": ["sql"], "criticality": 0.9, "value": 9.0,
            "cves": [{"id": "CVE-SIM-0501", "cvss": 6.8}],
        },
    }
    links = [
        ("Internet", "WebServer"),
        ("Internet", "MailServer"),
        ("WebServer", "AppServer"),
        ("MailServer", "Workstation1"),
        ("MailServer", "Workstation2"),
        ("Workstation1", "FileServer"),
        ("Workstation2", "FileServer"),
        ("Workstation2", "DomainController"),
        ("AppServer", "Database"),
        ("FileServer", "DomainController"),
        ("DomainController", "Database"),
        ("Workstation1", "Workstation2"),
    ]
    return build_graph(hosts, links)


def generate_topology(
    n_workstations: int = 12,
    n_servers: int = 4,
    n_dmz: int = 2,
    n_databases: int = 2,
    link_prob: float = 0.3,
    cve_prob: float = 0.7,
    seed: Optional[int] = 0,
) -> nx.DiGraph:
    """Generate a layered enterprise network: Internet -> DMZ -> workstations -> servers -> DC -> databases.

    Every host is reachable from the Internet, so the attack graph is connected.
    """
    rng = random.Random(seed)
    cve_ids = iter(range(1, 10_000))

    def cves() -> List[Dict]:
        if rng.random() > cve_prob:
            return []
        return [{"id": f"CVE-SIM-{next(cve_ids):04d}", "cvss": round(rng.uniform(4.0, 10.0), 1)}
                for _ in range(rng.randint(1, 2))]

    hosts: Dict[str, Dict] = {"Internet": {"type": "internet", "criticality": 0.0}}
    dmz = [f"DMZ{i + 1}" for i in range(n_dmz)]
    workstations = [f"WS{i + 1}" for i in range(n_workstations)]
    servers = [f"SRV{i + 1}" for i in range(n_servers)]
    databases = [f"DB{i + 1}" for i in range(n_databases)]

    for h in dmz:
        hosts[h] = {"type": "dmz", "services": ["https"], "criticality": 0.5, "value": 2.0, "cves": cves()}
    for h in workstations:
        hosts[h] = {"type": "workstation", "services": ["smb"], "criticality": 0.1, "value": 1.0,
                    "cves": cves(), "admin_credentials": rng.random() < 0.15}
    for h in servers:
        hosts[h] = {"type": "server", "services": ["http"], "criticality": 0.7, "value": 4.0, "cves": cves()}
    hosts["DC"] = {"type": "domain_controller", "services": ["ldap", "kerberos"], "criticality": 1.0,
                   "value": 10.0, "cves": cves(), "admin_credentials": True}
    for h in databases:
        hosts[h] = {"type": "database", "services": ["sql"], "criticality": 0.9, "value": 9.0, "cves": cves()}

    links: List[Tuple[str, str]] = [("Internet", h) for h in dmz]

    def connect(sources: Sequence[str], targets: Sequence[str]) -> None:
        for t in targets:
            others = [s for s in sources if s != t]
            links.append((rng.choice(others), t))  # guarantees reachability
            for s in others:
                if (s, t) not in links and rng.random() < link_prob:
                    links.append((s, t))

    connect(dmz, workstations)
    connect(workstations, workstations[1:])
    connect(workstations + dmz, servers)
    connect(servers + workstations, ["DC"])
    connect(servers + ["DC"], databases)
    return build_graph(hosts, [(u, v) for u, v in links if u != v])


def critical_assets(graph: nx.DiGraph, min_value: float = 9.0) -> List[str]:
    return [n for n, d in graph.nodes(data=True) if d.get("value", 0.0) >= min_value]

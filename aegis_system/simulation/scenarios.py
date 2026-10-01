"""Scenario networks for the closed-loop runs.

A scenario file (aegis_system/scenarios/<name>.json) describes a network: hosts,
links, the entry point and the critical targets. `make_environment` builds the
Phase 0 NetworkEnvironment on that network, so the attack itself is the
environment's standard kill-chain attacker. The file's "steps" list documents the
real-world campaign the network is modelled on; it is not executed here.

"random" builds a generated enterprise topology instead of reading a file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import networkx as nx

from ..core.environment import NetworkEnvironment
from ..search.graph_builder import build_graph, critical_assets, generate_topology

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"


@dataclass
class ScenarioSpec:
    name: str
    title: str
    description: str
    graph: nx.DiGraph
    entry: str
    targets: List[str]


def available_scenarios() -> List[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.json")) + ["random"]


def load_scenario(name: str, seed: int = 0) -> ScenarioSpec:
    if name == "random":
        graph = generate_topology(n_workstations=8, n_servers=3, seed=seed)
        return ScenarioSpec("random", "Generated enterprise network", f"generate_topology(seed={seed})",
                            graph, "Internet", critical_assets(graph))
    path = SCENARIO_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No scenario {name!r}; available: {', '.join(available_scenarios())}")
    data = json.loads(path.read_text())
    graph = build_graph(data["hosts"], [tuple(link) for link in data["links"]])
    for t in data["targets"]:
        if t not in graph:
            raise ValueError(f"Target {t!r} is not a host in {name}")
    return ScenarioSpec(data["name"], data.get("title", name), data.get("description", ""), graph,
                        data["entry"], list(data["targets"]))


LOOKALIKE_RATE = 0.05  # 5% of benign flows look like an attack phase (scans, admin sessions, backups)


def make_environment(name: str, seed: int = 0, p_fail: float = 0.05, lookalike_rate: float = LOOKALIKE_RATE,
                     **kwargs) -> NetworkEnvironment:
    spec = load_scenario(name, seed)
    return NetworkEnvironment(spec.graph, entry=spec.entry, targets=spec.targets, seed=seed, p_fail=p_fail,
                              lookalike_rate=lookalike_rate, **kwargs)


__all__ = ["SCENARIO_DIR", "ScenarioSpec", "available_scenarios", "load_scenario", "make_environment"]

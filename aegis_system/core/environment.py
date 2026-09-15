"""Simulated network the agent senses and acts on.

Nothing here touches a real network. Defensive actions change the simulated
state and return the iptables command that *would* be run, as a string.

Each tick the simulated attacker either advances one kill-chain phase or, once
in the Exploitation or C2 phase, tries to move one host closer to a critical
asset. `observe()` returns noisy flow features generated from the attacker's
true phase, so later phases have something realistic to classify and filter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from ..search.graph_builder import critical_assets, example_topology
from .types import ExecutionResult

KILL_CHAIN = ["Recon", "Weaponization", "Delivery", "Exploitation", "C2", "Exfiltration"]
LATERAL_PHASES = {"Exploitation", "C2"}

FEATURES = ["duration", "bytes_out", "bytes_in", "packets", "syn_ratio", "distinct_ports"]

# Mean feature vector per traffic profile; noise is added per flow.
PROFILES: Dict[str, Tuple[float, ...]] = {
    "benign":        (2.0,  1_500.0, 8_000.0,  20.0, 0.05,  1.0),
    "Recon":         (0.1,     60.0,     0.0,   2.0, 0.95, 40.0),
    "Weaponization": (1.0,    800.0,   900.0,  10.0, 0.20,  2.0),
    "Delivery":      (3.0,  9_000.0, 1_000.0,  40.0, 0.10,  1.0),
    "Exploitation":  (0.8,  2_500.0, 2_500.0,  30.0, 0.30,  3.0),
    "C2":            (30.0,   300.0,   300.0,  60.0, 0.02,  1.0),
    "Exfiltration":  (20.0, 90_000.0,  500.0, 300.0, 0.02,  1.0),
}

ACTIONS = {"isolate", "restore", "block_edge", "unblock_edge", "kill_process", "patch"}


@dataclass(frozen=True)
class ActionCommand:
    name: str
    target: str
    destination: Optional[str] = None  # only for block_edge / unblock_edge

    def __str__(self) -> str:
        if self.destination is not None:
            return f"{self.name}({self.target}, {self.destination})"
        return f"{self.name}({self.target})"


@dataclass
class Flow:
    src: str
    dst: str
    features: Dict[str, float]
    malicious: bool  # ground truth; the agent must not read this


@dataclass
class Observation:
    tick: int
    flows: List[Flow]

    def feature_matrix(self) -> np.ndarray:
        return np.array([[f.features[k] for k in FEATURES] for f in self.flows])


@dataclass
class EnvironmentState:
    tick: int = 0
    phase_index: int = 0
    compromised: Set[str] = field(default_factory=set)
    persistent: Set[str] = field(default_factory=set)
    isolated: Set[str] = field(default_factory=set)
    patched: Set[str] = field(default_factory=set)
    blocked: Set[Tuple[str, str]] = field(default_factory=set)
    attacker_position: Optional[str] = None

    @property
    def phase(self) -> str:
        return KILL_CHAIN[self.phase_index]


def iptables_command(action: ActionCommand) -> str:
    """The command a real deployment would run. Returned as text, never executed."""
    if action.name == "isolate":
        return (f"iptables -I FORWARD -s {action.target} -j DROP && "
                f"iptables -I FORWARD -d {action.target} -j DROP")
    if action.name == "restore":
        return (f"iptables -D FORWARD -s {action.target} -j DROP && "
                f"iptables -D FORWARD -d {action.target} -j DROP")
    if action.name == "block_edge":
        return f"iptables -I FORWARD -s {action.target} -d {action.destination} -j DROP"
    if action.name == "unblock_edge":
        return f"iptables -D FORWARD -s {action.target} -d {action.destination} -j DROP"
    if action.name == "kill_process":
        return f"edr kill-suspicious --host {action.target}"
    if action.name == "patch":
        return f"patch-manager apply --host {action.target}"
    raise ValueError(f"Unknown action {action.name!r}")


class NetworkEnvironment:
    def __init__(
        self,
        graph: Optional[nx.DiGraph] = None,
        entry: str = "Internet",
        targets: Optional[List[str]] = None,
        seed: Optional[int] = 0,
        p_fail: float = 0.1,
        attacker_success: float = 0.6,
        phase_advance: float = 0.5,
        persistence_prob: float = 0.3,
        benign_flows: int = 8,
    ):
        self.graph = graph if graph is not None else example_topology()
        if entry not in self.graph:
            raise ValueError(f"Entry node {entry!r} is not in the graph")
        self.entry = entry
        self.targets = list(targets) if targets is not None else critical_assets(self.graph)
        self.seed = seed
        self.p_fail = p_fail
        self.attacker_success = attacker_success
        self.phase_advance = phase_advance
        self.persistence_prob = persistence_prob
        self.benign_flows = benign_flows
        self.reset()

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> EnvironmentState:
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.state = EnvironmentState(attacker_position=self.entry)
        self.events: List[Dict] = []
        return self.state

    def step(self) -> EnvironmentState:
        """Advance the simulation by one tick (attacker acts)."""
        s = self.state
        s.tick += 1
        if s.phase in LATERAL_PHASES:
            self._attempt_lateral_move()
        elif s.phase_index < KILL_CHAIN.index("Exploitation") and self.rng.random() < self.phase_advance:
            s.phase_index += 1
            self._log("phase", phase=s.phase)
        if self.breached and s.phase != "Exfiltration":
            s.phase_index = KILL_CHAIN.index("Exfiltration")
            self._log("phase", phase=s.phase)
        return s

    def run(self, ticks: int) -> List[Observation]:
        observations = []
        for _ in range(ticks):
            self.step()
            observations.append(self.observe())
        return observations

    # ------------------------------------------------------------------ attacker

    def usable_graph(self) -> nx.DiGraph:
        """The topology minus isolated hosts and blocked edges."""
        s = self.state
        view = nx.subgraph_view(
            self.graph,
            filter_node=lambda n: n not in s.isolated,
            filter_edge=lambda u, v: (u, v) not in s.blocked,
        )
        return view

    def footholds(self) -> Set[str]:
        s = self.state
        return {h for h in s.compromised | {self.entry} if h not in s.isolated}

    def _next_hop(self) -> Optional[Tuple[str, str]]:
        g = self.usable_graph()
        open_targets = [t for t in self.targets if t not in self.state.compromised and t in g]
        best: Optional[Tuple[float, List[str]]] = None
        for source in self.footholds():
            if source not in g:
                continue
            lengths, paths = nx.single_source_dijkstra(g, source, weight="weight")
            for t in open_targets:
                if t in lengths and (best is None or lengths[t] < best[0]):
                    best = (lengths[t], paths[t])
        if best is None or len(best[1]) < 2:
            return None
        path = best[1]
        # first hop that is not already compromised
        for u, v in zip(path, path[1:]):
            if v not in self.state.compromised:
                return u, v
        return None

    def _attempt_lateral_move(self) -> None:
        hop = self._next_hop()
        if hop is None:
            self._log("stalled")
            return
        u, v = hop
        s = self.state
        exploit_p = self.attacker_success * (0.5 if v in s.patched else 1.0)
        if self.rng.random() < exploit_p:
            s.compromised.add(v)
            s.attacker_position = v
            if self.rng.random() < self.persistence_prob:
                s.persistent.add(v)
            if s.phase == "Exploitation":
                s.phase_index = KILL_CHAIN.index("C2")
            self._log("compromise", src=u, dst=v)
        else:
            self._log("exploit_failed", src=u, dst=v)

    # ------------------------------------------------------------------ sensing

    def observe(self) -> Observation:
        flows: List[Flow] = []
        edges = list(self.usable_graph().edges())
        for _ in range(self.benign_flows):
            if not edges:
                break
            u, v = self.rng.choice(edges)
            flows.append(Flow(u, v, self._sample("benign"), malicious=False))
        attacker = self.state.attacker_position
        if attacker is not None and attacker not in self.state.isolated:
            succ = [v for v in self.usable_graph().successors(attacker)] or [attacker]
            dst = self.rng.choice(succ)
            flows.append(Flow(attacker, dst, self._sample(self.state.phase), malicious=True))
        self.rng.shuffle(flows)
        return Observation(self.state.tick, flows)

    def _sample(self, profile: str) -> Dict[str, float]:
        means = np.array(PROFILES[profile])
        noisy = self.np_rng.normal(means, 0.25 * means + 0.01)
        noisy[4] = np.clip(noisy[4], 0.0, 1.0)  # syn_ratio is a fraction
        return {k: float(max(0.0, x)) for k, x in zip(FEATURES, noisy)}

    def ground_truth(self) -> Dict:
        s = self.state
        return {
            "tick": s.tick,
            "phase": s.phase,
            "compromised": sorted(s.compromised),
            "persistent": sorted(s.persistent),
            "breached": self.breached,
            "contained": self.is_contained(),
        }

    # ------------------------------------------------------------------ actuation

    def apply(self, action: ActionCommand) -> ExecutionResult:
        """Apply a defensive action. It fails with probability p_fail."""
        if action.name not in ACTIONS:
            raise ValueError(f"Unknown action {action.name!r}; expected one of {sorted(ACTIONS)}")
        if action.target not in self.graph:
            raise ValueError(f"Unknown host {action.target!r}")
        if action.destination is not None and not self.graph.has_edge(action.target, action.destination):
            raise ValueError(f"No link {action.target!r} -> {action.destination!r}")

        command = iptables_command(action)
        success = self.rng.random() >= self.p_fail
        s = self.state
        if success:
            if action.name == "isolate":
                s.isolated.add(action.target)
            elif action.name == "restore":
                s.isolated.discard(action.target)
            elif action.name == "block_edge":
                s.blocked.add((action.target, action.destination))
            elif action.name == "unblock_edge":
                s.blocked.discard((action.target, action.destination))
            elif action.name == "kill_process":
                if action.target in s.persistent:
                    success = False  # attacker keeps a foothold
                else:
                    s.compromised.discard(action.target)
            elif action.name == "patch":
                s.patched.add(action.target)
        self._log("action", action=str(action), success=success)
        return ExecutionResult(
            step=action,
            success=success,
            observed_state={"compromised": sorted(s.compromised), "isolated": sorted(s.isolated)},
            command=command,
        )

    # ------------------------------------------------------------------ status

    @property
    def breached(self) -> bool:
        return any(t in self.state.compromised for t in self.targets)

    def is_contained(self) -> bool:
        """True when no live compromised host can reach an uncompromised target."""
        g = self.usable_graph()
        live = [h for h in self.state.compromised if h in g]
        open_targets = [t for t in self.targets if t not in self.state.compromised and t in g]
        return not any(nx.has_path(g, h, t) for h in live for t in open_targets)

    def _log(self, kind: str, **data) -> None:
        self.events.append({"tick": self.state.tick, "event": kind, **data})

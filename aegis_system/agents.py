"""Four defender agents with one interface: act(percept) -> actions taken.

    ReflexAgent      signature rules only: decision-tree rules in the production system,
                     isolate the source of any flow they classify as an attack
    ModelBasedAgent  keeps a belief state (kill-chain HMM + per-host DBN) and isolates
                     hosts whose P(compromised) crosses a threshold
    GoalBasedAgent   the full A.E.G.I.S. pipeline: beliefs -> grounding -> proofs -> goal
                     test (BFS: no path from a compromised host to a critical one) ->
                     min-cut choke points -> POP plan -> monitored execution
    UtilityAgent     chooses actions with the highest EU(a) = sum_s P(s) U(s, a), where U
                     is the loss avoided minus the downtime cost

Agents see only what a defender would: the flows of the tick, their own firewall
configuration, and the results of their own actions. They act through the
environment's `apply`, the actuator. Each agent records its reasoning in the
TraceStore, so every action can be explained back to the detection behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .core.environment import FEATURES, ActionCommand, Flow, NetworkEnvironment
from .core.logic import Atom, Const
from .core.trace import TraceStore
from .core.types import Alert, BeliefState, ExecutionResult, Proof
from .knowledge.grounding import Grounder
from .knowledge.production_system import ProductionSystem
from .perception.classifier import (
    AttackClassifier, extract_rules, flow_facts_for, indicator_names,
)
from .perception.datasets import synthetic_flows
from .perception.regressor import TimeToCompromiseRegressor, generate_ttc_dataset, host_features
from .planning.domain import defense_problem
from .planning.executor import Executor
from .probabilistic.dbn import apt_dbn
from .probabilistic.hmm_engine import kill_chain_hmm
from .search.adversarial_search import NetworkGame, best_defense
from .search.path_search import astar

from sklearn.tree import DecisionTreeClassifier

# Later kill-chain categories first: the tick's HMM observation is its most advanced alert.
SEVERITY = ["dos", "u2r", "r2l", "probe", "normal"]


@dataclass
class Percept:
    tick: int
    flows: List[Flow]
    isolated: Set[str]
    blocked: Set[Tuple[str, str]]
    feedback: List[ExecutionResult] = field(default_factory=list)

    @classmethod
    def from_environment(cls, env: NetworkEnvironment, flows: List[Flow],
                         feedback: Sequence[ExecutionResult] = ()) -> "Percept":
        return cls(env.state.tick, flows, set(env.state.isolated), set(env.state.blocked), list(feedback))


# --------------------------------------------------------------------------- shared components


class Perception:
    """Decision-tree detector trained on simulated flows of the same network (the Phase 6 1-SE choice)."""

    def __init__(self, env: NetworkEnvironment, seed: int = 0, training_ticks: int = 600, min_confidence: float = 0.6):
        train_env = NetworkEnvironment(env.graph, entry=env.entry, targets=env.targets, seed=seed + 1000,
                                       lookalike_rate=env.lookalike_rate)
        data = synthetic_flows(training_ticks, seed=seed + 1000, env=train_env, restart_every=30)
        cut = int(0.8 * len(data))
        train, held_out = data.iloc[:cut], data.iloc[cut:]
        self.classifier = AttackClassifier(DecisionTreeClassifier(max_depth=8, random_state=seed)).fit(
            train[FEATURES], train["label"])
        self.entry = env.entry
        self.min_confidence = min_confidence
        # Detector false-positive rate on held-out benign flows: the DBN sensor model's false-alarm rate.
        benign = held_out[held_out["label"] == "normal"]
        proba = self.classifier.predict_proba(benign[FEATURES])
        classes = self.classifier.classes_
        normal = classes.index("normal")
        flagged = (proba.argmax(axis=1) != normal) & (proba.max(axis=1) >= min_confidence)
        self.false_alarm_rate = max(float(flagged.mean()), 1e-3)  # floor: never assume a perfect detector

    def alerts(self, flows: Sequence[Flow], tick: int, trace: Optional[TraceStore]) -> List[Alert]:
        if not flows:
            return []
        frame = pd.DataFrame([{**f.features, "src": f.src, "dst": f.dst} for f in flows])
        alerts = [a for a in self.classifier.alerts(frame, FEATURES, min_confidence=self.min_confidence)
                  if a.host != self.entry]
        for a in alerts:
            if trace is not None:
                a.trace_id = trace.record("ml", f"t={tick} {a.attack_class} from {a.host} (p={a.confidence:.2f})",
                                          {"tick": tick, "host": a.host, "class": a.attack_class,
                                           "confidence": a.confidence})
        return alerts


class BeliefTracker:
    """Kill-chain phase (HMM, filtered online) and per-host compromise (one two-slice DBN per host)."""

    def __init__(self, hosts: Sequence[str], alert_rate: float = 0.6, false_alarm: float = 0.02):
        self.hmm = kill_chain_hmm()
        self.dbn = apt_dbn(alert_rate=alert_rate, false_alarm=false_alarm)
        self.hosts = list(hosts)
        self.phase: Optional[np.ndarray] = None
        self.host_belief = {h: None for h in self.hosts}

    def update(self, alerts: Sequence[Alert], tick: int, trace: Optional[TraceStore],
               unobserved: Sequence[str] = ()) -> BeliefState:
        """unobserved: hosts whose traffic cannot be seen (isolated), so silence is no evidence."""
        seen = {a.attack_class for a in alerts}
        observation = next(c for c in SEVERITY if c in seen or c == "normal")
        self.phase = self.hmm.step(self.phase, observation)
        alerted = {a.host for a in alerts}
        probs = {}
        for h in self.hosts:
            b = self.host_belief[h]
            evidence = {} if h in unobserved else {"Alert": h in alerted}
            b = self.dbn.filter_step(self.dbn.initial_belief() if b is None else b, evidence, predict=b is not None)
            self.host_belief[h] = b
            probs[h] = self.dbn.marginal(b, "Foothold")
        posterior = {s: float(p) for s, p in zip(self.hmm.states, self.phase)}
        tid = None
        if trace is not None:
            phase = max(posterior, key=posterior.get)
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            tid = trace.record("belief", f"t={tick} phase={phase} ({posterior[phase]:.2f}); "
                                         + ", ".join(f"P({h})={p:.2f}" for h, p in top),
                               {"tick": tick, "phase_posterior": posterior, "host_compromise_prob": probs,
                                "hmm_observation": observation},
                               [a.trace_id for a in alerts])
        return BeliefState(posterior, probs, tid)


# --------------------------------------------------------------------------- agents


class Agent:
    name = "agent"
    uses_belief = True

    def __init__(self, env: NetworkEnvironment, trace: Optional[TraceStore] = None, seed: int = 0,
                 perception: Optional[Perception] = None):
        self.env = env  # used only as the actuator (apply) and for the static topology
        self.graph = env.graph
        self.entry = env.entry
        self.targets = list(env.targets)
        self.hosts = [h for h in self.graph.nodes if h != self.entry]
        self.trace = trace
        self.perception = perception or Perception(env, seed)
        self.beliefs = BeliefTracker(self.hosts, false_alarm=self.perception.false_alarm_rate)
        self.last_belief: Optional[BeliefState] = None
        self.proof_depths: List[int] = []

    def act(self, percept: Percept) -> List[ExecutionResult]:
        raise NotImplementedError

    def _sense(self, percept: Percept) -> Tuple[List[Alert], BeliefState]:
        alerts = self.perception.alerts(percept.flows, percept.tick, self.trace)
        belief = self.beliefs.update(alerts, percept.tick, self.trace, unobserved=percept.isolated)
        self.last_belief = belief
        return alerts, belief

    def _apply(self, command: ActionCommand, reason: str, parents: Sequence[Optional[str]], tick: int,
               payload: Optional[Dict] = None) -> ExecutionResult:
        result = self.env.apply(command)
        if self.trace is not None:
            result.trace_id = self.trace.record(
                "act", f"t={tick} {command} -> {'ok' if result.success else 'FAILED'}: {reason}",
                {"tick": tick, "command": result.command, "success": result.success, **(payload or {})}, parents)
        return result

    def criticality(self, h: str) -> float:
        return float(self.graph.nodes[h].get("criticality", 0.0))


class ReflexAgent(Agent):
    """Condition-action rules over single flows; no memory between ticks."""

    name = "reflex"
    uses_belief = False

    def __init__(self, env, trace=None, seed=0, perception=None):
        super().__init__(env, trace, seed, perception)
        pipeline = self.perception.classifier.pipeline
        self.rules = extract_rules(pipeline)
        self.indicators = indicator_names(pipeline)

    def act(self, percept: Percept) -> List[ExecutionResult]:
        ps = ProductionSystem(self.rules)  # fresh working memory every tick: a pure reflex
        for i, flow in enumerate(percept.flows):
            for fact in flow_facts_for(f"f{i}", flow.features, self.indicators):
                ps.assert_fact(fact)
        ps.run(10_000)
        results = []
        acted: Set[str] = set()
        for fact in ps.facts("Classified"):
            label = fact.args[1].name
            i = int(fact.args[0].name[1:])
            src = percept.flows[i].src
            if label == "normal" or src == self.entry or src in percept.isolated or src in acted:
                continue
            acted.add(src)
            tid = None
            if self.trace is not None:
                tid = self.trace.record("rule", f"t={percept.tick} signature rule: flow from {src} is {label}",
                                        {"tick": percept.tick, "host": src, "class": label})
            results.append(self._apply(ActionCommand("isolate", src), f"signature match ({label})", [tid],
                                       percept.tick))
        return results


class ModelBasedAgent(Agent):
    """Isolates hosts whose belief of compromise crosses the threshold."""

    name = "model_based"

    def __init__(self, env, trace=None, seed=0, perception=None, threshold: float = 0.8):
        super().__init__(env, trace, seed, perception)
        self.threshold = threshold

    def act(self, percept: Percept) -> List[ExecutionResult]:
        _, belief = self._sense(percept)
        results = []
        for h, p in sorted(belief.host_compromise_prob.items(), key=lambda kv: -kv[1]):
            if p >= self.threshold and h not in percept.isolated:
                results.append(self._apply(ActionCommand("isolate", h), f"P(compromised)={p:.2f}",
                                           [belief.trace_id], percept.tick, {"probability": p}))
        return results


class UtilityAgent(Agent):
    """Greedy expected-utility maximizer over isolations and edge blocks.

    States s: each suspected host is compromised or not (independently, with the
    belief's probabilities). For an action a on suspected host h,
        U(compromised, a) = value at risk removed by a - cost(a),   U(clean, a) = -cost(a)
    so EU(a) = P(h) * risk_removed(a) - cost(a); doing nothing has EU 0.
    """

    name = "utility"

    def __init__(self, env, trace=None, seed=0, perception=None, min_probability: float = 0.1,
                 downtime_weight: float = 4.0, block_cost: float = 0.3, max_actions: int = 3):
        super().__init__(env, trace, seed, perception)
        self.min_probability = min_probability
        self.downtime_weight = downtime_weight
        self.block_cost = block_cost
        self.max_actions = max_actions

    def _usable(self, isolated: Set[str], blocked: Set[Tuple[str, str]]) -> nx.DiGraph:
        g = self.graph.copy()
        g.remove_nodes_from(isolated)
        g.remove_edges_from(blocked)
        return g

    def value_at_risk(self, g: nx.DiGraph, h: str, at_risk: Sequence[str]) -> float:
        if h not in g:
            return 0.0
        reach = nx.descendants(g, h) | {h}
        return sum(self.graph.nodes[t].get("value", 0.0) for t in at_risk if t in reach)

    def act(self, percept: Percept) -> List[ExecutionResult]:
        _, belief = self._sense(percept)
        probs = belief.host_compromise_prob
        isolated, blocked = set(percept.isolated), set(percept.blocked)
        at_risk = [t for t in self.targets if probs.get(t, 0.0) < 0.5]
        results = []
        for _ in range(self.max_actions):
            g = self._usable(isolated, blocked)
            suspects = [h for h in self.hosts if probs.get(h, 0.0) >= self.min_probability and h not in isolated]
            options: List[Tuple[float, ActionCommand, Dict]] = []
            for h in suspects:
                p, risk = probs[h], self.value_at_risk(g, h, at_risk)
                cost = self.downtime_weight * self.criticality(h) + 0.5
                options.append((p * risk - cost, ActionCommand("isolate", h),
                                {"p": p, "risk_removed": risk, "cost": cost}))
                for v in g.successors(h):
                    g2 = g.copy()
                    g2.remove_edge(h, v)
                    removed = risk - self.value_at_risk(g2, h, at_risk)
                    options.append((p * removed - self.block_cost, ActionCommand("block_edge", h, v),
                                    {"p": p, "risk_removed": removed, "cost": self.block_cost}))
            if not options:
                break
            eu, command, detail = max(options, key=lambda o: (o[0], str(o[1])))
            if eu <= 0:
                break
            results.append(self._apply(command, f"EU={eu:.2f}", [belief.trace_id], percept.tick,
                                       {"expected_utility": eu, **detail}))
            if results[-1].success:
                if command.name == "isolate":
                    isolated.add(command.target)
                else:
                    blocked.add((command.target, command.destination))
        return results


class GoalBasedAgent(Agent):
    """The full A.E.G.I.S. pipeline. Goal: no path from a compromised host to an uncompromised critical one."""

    name = "goal_based"

    def __init__(self, env, trace=None, seed=0, perception=None, ground_threshold: float = 0.5,
                 isolate_below_criticality: float = 0.9):
        super().__init__(env, trace, seed, perception)
        self.grounder = Grounder(self.graph, trace=trace, threshold=ground_threshold)
        self.ground_threshold = ground_threshold
        self.isolate_below = isolate_below_criticality
        self.proven: Dict[str, Proof] = {}
        self.grounded: Set[Tuple[str, str]] = set()
        self.ttc = TimeToCompromiseRegressor(seed).fit(*generate_ttc_dataset(800, seed=seed))
        self.centrality = nx.betweenness_centrality(self.graph)
        self.game = NetworkGame(self.graph, self.entry, self.targets)
        self.recent_alerts: List[int] = []

    # ----------------------------------------------------------------- think

    def _ground(self, alerts: Sequence[Alert], belief: BeliefState) -> None:
        for a in alerts:
            key = (a.host, a.attack_class)
            if key in self.grounded or belief.host_compromise_prob.get(a.host, 0.0) < self.ground_threshold:
                continue
            if self.grounder.ground(a, belief).atoms:
                self.grounded.add(key)

    def _prove(self) -> None:
        kb = self.grounder.kb
        for h in self.hosts:
            if h in self.proven or not kb.ask(f"Compromised({h})", method="backward", record=False).proved:
                continue
            proof = kb.ask(f"Compromised({h})", method="resolution")
            if proof.proved:
                self.proven[h] = proof
                self.proof_depths.append(len(proof.steps))

    def _usable(self, percept: Percept) -> nx.DiGraph:
        g = self.graph.copy()
        g.remove_nodes_from(percept.isolated)
        g.remove_edges_from(percept.blocked)
        return g

    def goal_violations(self, percept: Percept) -> List[Tuple[str, str]]:
        """(compromised host, critical host) pairs still connected: the goal test, by BFS.

        A live compromised critical host is a path of length zero to a critical
        host (itself), so it violates the goal too: it can still be read and exfiltrated.
        """
        g = self._usable(percept)
        open_targets = [t for t in self.targets if t not in self.proven]
        pairs = []
        for h in sorted(self.proven):
            if h not in g:
                continue
            if h in self.targets:
                pairs.append((h, h))
            reach = nx.descendants(g, h)  # breadth-first reachability
            pairs += [(h, t) for t in open_targets if t in reach]
        return pairs

    def _choke(self, g: nx.DiGraph, sources: Sequence[str], targets: Sequence[str]) -> Set[Tuple[str, str]]:
        """Minimum edge cut separating every source from every target (super-source and super-sink)."""
        h = nx.DiGraph()
        h.add_edges_from(g.edges(), capacity=1)
        src, sink = ("__src__",), ("__sink__",)
        for s in sources:
            h.add_edge(src, s)
        for t in targets:
            h.add_edge(t, sink)
        if not nx.has_path(h, src, sink):
            return set()
        _, (reachable, _) = nx.minimum_cut(h, src, sink)
        return {(u, v) for u, v in g.edges() if u in reachable and v not in reachable}

    # ----------------------------------------------------------------- act

    def act(self, percept: Percept) -> List[ExecutionResult]:
        alerts, belief = self._sense(percept)
        self.recent_alerts = (self.recent_alerts + [len(alerts)])[-5:]
        self._ground(alerts, belief)
        self._prove()
        violations = self.goal_violations(percept)
        if not violations:
            return []

        live = sorted({h for h, _ in violations})
        # Isolate cheap hosts and compromised targets (isolation is the only way to stop exfiltration
        # from a target); for other critical hosts, cut their paths instead.
        isolate = [h for h in live if self.criticality(h) < self.isolate_below or h in self.targets]
        rest = [h for h in live if h not in isolate]
        g = self._usable(percept)
        g.remove_nodes_from(isolate)
        open_targets = [t for t in self.targets if t not in self.proven]
        cut = self._choke(g, rest, open_targets) if rest else set()
        goal = {Atom("Isolated", (Const(h),)) for h in isolate} | \
               {Atom("Blocked", (Const(u), Const(v))) for u, v in cut}
        if not goal:
            return []

        threatened = min(open_targets or live, key=lambda t: nx.shortest_path_length(self.graph, live[0], t)
                         if nx.has_path(self.graph, live[0], t) else 1e9)
        ttc = self.ttc.predict_host(host_features(self.graph, threatened, patch_lag=30.0,
                                                  alert_rate=float(np.mean(self.recent_alerts)),
                                                  centrality=self.centrality))
        path = astar(self._usable(percept), live[0], open_targets) if open_targets else None
        game_value = best_defense(self.game, live, depth=2).minimax_value
        defense_tid = None
        if self.trace is not None:
            defense_tid = self.trace.record(
                "defense",
                f"t={percept.tick} contain {', '.join(live)}: isolate {isolate or '-'}, block {sorted(cut) or '-'}",
                {"tick": percept.tick, "violations": violations, "isolate": isolate, "cut": sorted(cut),
                 "attack_path": path.path if path else [], "ttc_hours": ttc, "minimax_value": game_value},
                [self.proven[h].trace_id for h in live] + [belief.trace_id])

        facts = [Atom("Isolated", (Const(h),)) for h in percept.isolated] + \
                [Atom("Blocked", (Const(u), Const(v))) for u, v in percept.blocked]
        problem = defense_problem(self.graph, sorted(self.proven), goal, ttc_hours=ttc, extra_facts=facts,
                                  entry=self.entry)
        report = Executor(self.env, max_replans=2, trace=self.trace).execute(problem, parent_trace=defense_tid)
        return report.results


AGENTS = {cls.name: cls for cls in (ReflexAgent, ModelBasedAgent, GoalBasedAgent, UtilityAgent)}

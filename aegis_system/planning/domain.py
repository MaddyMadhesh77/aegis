"""The incident-response planning domain and its link to the simulated network.

Objects are host names. Static facts come from the topology:
Host(h) for every host except the entry point, Link(u, v) for every edge, and
Vulnerable(h) for hosts with a known CVE. Dynamic facts are Compromised(h),
Isolated(h), Blocked(u, v), MemoryCaptured(h), Patched(h), Clean(h), Restored(h).

Two ways to clean a host trade money against time, which is what urgency
(from the predicted time-to-compromise) decides between:

    slow, cheap:  DumpMemory(h) -> Isolate(h) -> Reimage(h)
    fast, costly: KillProcess(h)   (loses evidence; fails against persistence, after
                                    which the executor records Persistent(h))
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx

from ..core.environment import ActionCommand
from ..core.logic import Atom, Const
from .strips_planner import Action, ActionSchema, Problem, ground, urgency_from_ttc

SCHEMAS: List[ActionSchema] = [
    ActionSchema.parse("DumpMemory(h)", pre="Host(h) & Compromised(h) & ~Isolated(h) & ~MemoryCaptured(h)",
                       add="MemoryCaptured(h)", cost=1.0, duration=3.0),
    ActionSchema.parse("Isolate(h)", pre="Host(h) & Compromised(h) & ~Isolated(h)",
                       add="Isolated(h)", cost=1.0, duration=1.0),
    ActionSchema.parse("BlockEdge(u, v)", pre="Link(u, v) & ~Blocked(u, v)",
                       add="Blocked(u, v)", cost=0.5, duration=1.0),
    ActionSchema.parse("Reimage(h)", pre="Host(h) & Isolated(h) & MemoryCaptured(h) & Compromised(h)",
                       add="Clean(h)", delete="Compromised(h)", cost=2.0, duration=6.0),
    ActionSchema.parse("KillProcess(h)", pre="Host(h) & Compromised(h) & ~Isolated(h) & ~Persistent(h)",
                       add="Clean(h)", delete="Compromised(h)", cost=8.0, duration=1.0),
    ActionSchema.parse("Patch(h)", pre="Host(h) & Vulnerable(h) & ~Patched(h) & ~Compromised(h)",
                       add="Patched(h)", cost=1.0, duration=4.0),
    ActionSchema.parse("Restore(h)", pre="Host(h) & Isolated(h) & Clean(h)",
                       add="Restored(h)", delete="Isolated(h)", cost=0.5, duration=1.0),
]

# Predicates the executor can read back from the environment's sensors.
OBSERVABLE = {"Compromised", "Isolated", "Blocked", "Patched", "MemoryCaptured"}

# What a failed action teaches the agent: a KillProcess that leaves the host
# compromised means the attacker has persistence there.
FAILURE_FACTS = {"KillProcess": "Persistent"}

COMMAND_NAMES = {
    "Isolate": "isolate", "BlockEdge": "block_edge", "Reimage": "reimage", "KillProcess": "kill_process",
    "Patch": "patch", "Restore": "restore", "DumpMemory": "collect_memory",
}


def _c(*names: str) -> Tuple[Const, ...]:
    return tuple(Const(n) for n in names)


def static_facts(graph: nx.DiGraph, entry: str = "Internet") -> Set[Atom]:
    facts: Set[Atom] = set()
    for h, attrs in graph.nodes(data=True):
        if h == entry:
            continue
        facts.add(Atom("Host", _c(h)))
        if attrs.get("cves"):
            facts.add(Atom("Vulnerable", _c(h)))
    for u, v in graph.edges():
        facts.add(Atom("Link", _c(u, v)))
    return facts


def cost_function(graph: nx.DiGraph):
    """Isolating a host costs more the more critical it is (downtime)."""

    def cost(schema: ActionSchema, args: Tuple[Const, ...]) -> Tuple[float, float]:
        if schema.name == "Isolate":
            crit = graph.nodes[args[0].name].get("criticality", 0.0)
            return schema.cost + 4.0 * crit, schema.duration
        return schema.cost, schema.duration

    return cost


def defense_problem(
    graph: nx.DiGraph,
    compromised: Iterable[str],
    goal,
    ttc_hours: Optional[float] = None,
    extra_facts: Iterable[Atom] = (),
    entry: str = "Internet",
    schemas: Sequence[ActionSchema] = SCHEMAS,
) -> Problem:
    """A grounded planning problem for the current incident.

    goal is a formula string ("Isolated(WS1) & MemoryCaptured(WS1)") or a set of atoms
    (e.g. DefenseObjective.goal_atoms).
    """
    init = static_facts(graph, entry) | {Atom("Compromised", _c(h)) for h in compromised} | set(extra_facts)
    objects = list(graph.nodes())  # the entry is an object (its links can be blocked) but not a Host
    return ground(list(schemas), objects, init, goal, urgency_from_ttc(ttc_hours), cost_function(graph),
                  dynamic=FAILURE_FACTS.values())


def to_command(action: Action) -> ActionCommand:
    names = [a.name for a in action.args]  # type: ignore[union-attr]
    return ActionCommand(COMMAND_NAMES[action.name], names[0], names[1] if len(names) > 1 else None)


def observed_atoms(sensed: Mapping[str, List]) -> Set[Atom]:
    """Turn the environment's sensor report into atoms of the observable predicates."""
    out: Set[Atom] = set()
    for h in sensed.get("compromised", []):
        out.add(Atom("Compromised", _c(h)))
    for h in sensed.get("isolated", []):
        out.add(Atom("Isolated", _c(h)))
    for u, v in sensed.get("blocked", []):
        out.add(Atom("Blocked", _c(u, v)))
    for h in sensed.get("patched", []):
        out.add(Atom("Patched", _c(h)))
    for h in sensed.get("memory_captured", []):
        out.add(Atom("MemoryCaptured", _c(h)))
    return out


def merge_observation(problem: Problem, believed: Iterable[Atom], sensed: Mapping[str, List]) -> frozenset:
    """New belief: observable predicates replaced by what was sensed, the rest kept.

    When the report has a "scope", host compromise was only checked on those hosts,
    so Compromised(h) for hosts outside the scope is kept from the belief.
    """
    scope = sensed.get("scope")

    def keep(a: Atom) -> bool:
        if a.pred == "Compromised" and scope is not None:
            return a.args[0].name not in scope
        return a.pred not in OBSERVABLE

    kept = {a for a in problem.positive_view(frozenset(believed)) if keep(a)}
    return problem.complete(kept | observed_atoms(sensed))


def goal_text(atoms: Iterable[Atom]) -> str:
    return " & ".join(sorted(str(a) for a in atoms))


__all__ = [
    "COMMAND_NAMES", "FAILURE_FACTS", "OBSERVABLE", "SCHEMAS", "cost_function", "defense_problem", "goal_text", "merge_observation",
    "observed_atoms", "static_facts", "to_command",
]


def describe(problem: Problem) -> Dict[str, object]:
    return {"actions": len(problem.actions), "init": sorted(map(str, problem.positive_view(problem.init))),
            "goal": sorted(map(str, problem.goal))}

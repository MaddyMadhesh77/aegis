"""The closed loop: sense -> think -> plan -> act, tick by tick, with PEAS metrics.

Performance measures (the P of PEAS), computed from the environment's ground truth:

    time_to_containment   ticks from the first compromise until containment that lasts to the end
                          of the run: no live compromised host can reach an uncompromised target
                          and no target is held live (None if not contained at the end; 0 if
                          nothing was ever compromised or it was contained in the same tick)
    uptime                criticality-weighted share of host-ticks not spent isolated
    false_isolations      hosts the agent isolated that were never compromised
    proof_depth           mean resolution-proof length behind the agent's decisions (0 if none)
    breached              did the attacker ever hold a critical target
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .agents import AGENTS, Agent, Perception, Percept
from .core.environment import NetworkEnvironment
from .core.trace import TraceStore
from .simulation.scenarios import load_scenario, make_environment


@dataclass
class EpisodeResult:
    scenario: str
    agent: str
    seed: int
    metrics: Dict[str, object]
    timeline: List[Dict] = field(default_factory=list)
    trace: Dict[str, Dict] = field(default_factory=dict)
    topology: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> Dict:
        return {"scenario": self.scenario, "agent": self.agent, "seed": self.seed, "metrics": self.metrics,
                "timeline": self.timeline, "trace": self.trace, "topology": self.topology}

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.scenario}_{self.agent}_seed{self.seed}.json"
        path.write_text(json.dumps(self.to_json(), indent=1, default=str))
        return path


def contained(env: NetworkEnvironment) -> bool:
    s = env.state
    targets_held_live = any(t in s.compromised and t not in s.isolated for t in env.targets)
    return env.is_contained() and not targets_held_live


def topology_json(env: NetworkEnvironment) -> Dict[str, object]:
    return {"nodes": [{"id": n, **{k: v for k, v in d.items() if k != "cves"},
                       "cvss": max((c["cvss"] for c in d.get("cves", [])), default=0.0)}
                      for n, d in env.graph.nodes(data=True)],
            "edges": [list(e) for e in env.graph.edges()], "entry": env.entry, "targets": list(env.targets)}


def run_episode(
    agent_name: str,
    scenario: str = "solarwinds",
    ticks: int = 30,
    seed: int = 0,
    perception: Optional[Perception] = None,
    on_trace: Optional[Callable[[str, str, str], None]] = None,
) -> EpisodeResult:
    """Run one agent against one scenario. on_trace(tick, stage, summary) is called for every new trace entry."""
    env = make_environment(scenario, seed=seed)
    trace = TraceStore()
    agent: Agent = AGENTS[agent_name](env, trace=trace, seed=seed, perception=perception)
    hosts = [h for h in env.graph.nodes if h != env.entry]
    crit = {h: float(env.graph.nodes[h].get("criticality", 0.0)) for h in hosts}
    total_crit = sum(crit.values()) or 1.0

    timeline: List[Dict] = []
    ever_compromised: set = set()
    isolated_by_agent: set = set()
    downtime = 0.0
    first_compromise: Optional[int] = None
    last_uncontained: Optional[int] = None
    breached = False
    feedback: List = []
    seen = 0

    for _ in range(ticks):
        env.step()
        observation = env.observe()
        percept = Percept.from_environment(env, observation.flows, feedback)
        results = agent.act(percept)
        feedback = results
        s = env.state

        ever_compromised |= s.compromised
        breached = breached or any(t in s.compromised for t in env.targets)
        if first_compromise is None and s.compromised:
            first_compromise = s.tick
        if first_compromise is not None and not contained(env):
            last_uncontained = s.tick
        isolated_by_agent |= s.isolated  # only the agent isolates hosts
        downtime += sum(crit[h] for h in s.isolated if h in crit)

        belief = agent.last_belief
        timeline.append({
            "tick": s.tick,
            "phase": s.phase,
            "compromised": sorted(s.compromised),
            "isolated": sorted(s.isolated),
            "blocked": [list(e) for e in sorted(s.blocked)],
            "contained": contained(env),
            "alerts": sum(1 for n in trace.by_stage("ml") if n.payload.get("tick") == s.tick),
            "phase_posterior": belief.phase_posterior if belief else {},
            "host_compromise_prob": belief.host_compromise_prob if belief else {},
            "actions": [{"action": str(r.step), "command": r.command, "success": r.success,
                         "trace_id": r.trace_id} for r in results],
        })
        if on_trace is not None:
            nodes = sorted(trace.to_dict().items(), key=lambda kv: trace.get(kv[0]).seq)[seen:]
            for tid, n in nodes:
                on_trace(s.tick, n["stage"], n["summary"])
            seen += len(nodes)

    if first_compromise is None:
        ttc = 0
    elif not contained(env):
        ttc = None
    elif last_uncontained is None:
        ttc = 0  # contained within the tick of the first compromise
    else:
        ttc = last_uncontained + 1 - first_compromise
    metrics = {
        "time_to_containment": ttc,
        "contained_at_end": contained(env),
        "breached": breached,
        "uptime": 1.0 - downtime / (ticks * total_crit),
        "false_isolations": len(isolated_by_agent - ever_compromised),
        "isolations": len(isolated_by_agent),
        "blocked_edges": len(env.state.blocked),
        "actions": sum(len(t["actions"]) for t in timeline),
        "proof_depth": float(np.mean(agent.proof_depths)) if agent.proof_depths else 0.0,
        "compromised_hosts": len(ever_compromised),
    }
    return EpisodeResult(scenario, agent_name, seed, metrics, timeline, trace.to_dict(), topology_json(env))


def compare_agents(scenario: str = "solarwinds", seeds: Sequence[int] = range(5), ticks: int = 30,
                   agents: Sequence[str] = tuple(AGENTS)) -> List[Dict[str, object]]:
    """Every agent on the same incidents (same scenario and seeds); means of the PEAS measures."""
    rows = []
    for name in agents:
        runs = []
        for seed in seeds:
            env = make_environment(scenario, seed=seed)
            runs.append(run_episode(name, scenario, ticks, seed, perception=Perception(env, seed)).metrics)
        ttcs = [r["time_to_containment"] for r in runs if r["time_to_containment"] is not None]
        rows.append({
            "agent": name,
            "episodes": len(runs),
            "contained": sum(r["contained_at_end"] for r in runs),
            "mean_time_to_containment": float(np.mean(ttcs)) if ttcs else None,
            "breaches": sum(r["breached"] for r in runs),
            "mean_uptime": float(np.mean([r["uptime"] for r in runs])),
            "mean_false_isolations": float(np.mean([r["false_isolations"] for r in runs])),
            "mean_proof_depth": float(np.mean([r["proof_depth"] for r in runs])),
            "mean_actions": float(np.mean([r["actions"] for r in runs])),
        })
    return rows


def scenario_title(name: str) -> str:
    return load_scenario(name).title

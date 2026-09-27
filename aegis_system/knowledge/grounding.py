"""Grounding: turn a perception alert into logic atoms for the FOL knowledge base.

    alert (ML class on a host)  --semantic net-->  ATT&CK technique and tactic
                                --frames------->   host and incident frames filled in
                                --atoms-------->   Compromised(Host3), Uses(Attacker, T1003), ...

Grounding runs only when the posterior that the host is compromised is above a
threshold (from the BeliefState when there is one, otherwise the alert's own
confidence).

The topology contributes context facts through if_needed demons on the generic
host frame: Trusts(DC, h) for every domain member h that talks to a domain
controller, and HasAdminSession(h, DC) for hosts with cached admin credentials.

The if_added demon on privilege_level fires when a technique needs SYSTEM
(credential dumping does): it tells PrivEsc(host) to the knowledge base and
records a trace entry, so the proof can be explained back through the frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import networkx as nx

from ..core.logic import Atom, Const
from ..core.trace import TraceStore
from ..core.types import Alert, BeliefState, GroundedFacts
from ..reasoning.fol_engine import FOLKnowledgeBase
from .frame_system import Frame, FrameSystem
from .semantic_net import SemanticNet

ATTACKER = "Attacker"
PRIVILEGE_RANK = {"user": 0, "admin": 1, "SYSTEM": 2}
# Tactics whose observation means the host itself is under the attacker's control.
COMPROMISE_TACTICS = {"initial-access", "execution", "privilege-escalation", "credential-access",
                      "lateral-movement", "command-and-control", "exfiltration", "impact"}
GENERIC_FRAMES = {"workstation": "host", "server": "host", "dmz": "host", "database": "host",
                  "domain_controller": "server"}

INCIDENT_RULES = [
    # Lateral movement through trust (the Phase 1 rule).
    "forall x forall y forall z (Compromised(x) & Trusts(y, x) & TransfersAuth(x, y, z) -> Compromised(y))",
    # Dumped credentials of an admin session to d can be replayed against d.
    "forall h forall d (CredentialsDumped(h) & HasAdminSession(h, d) -> TransfersAuth(h, d, AdminCredential))",
    # Escalating to SYSTEM on a host means the host is compromised.
    "forall h (PrivEsc(h) -> Compromised(h))",
]


def _a(pred: str, *args: str) -> Atom:
    return Atom(pred, tuple(Const(x) for x in args))


class Grounder:
    def __init__(
        self,
        graph: nx.DiGraph,
        kb: Optional[FOLKnowledgeBase] = None,
        net: Optional[SemanticNet] = None,
        trace: Optional[TraceStore] = None,
        threshold: float = 0.5,
        rules: Sequence[str] = INCIDENT_RULES,
    ):
        self.graph = graph
        self.trace = trace
        self.kb = kb if kb is not None else FOLKnowledgeBase(trace)
        self.net = net if net is not None else SemanticNet.from_attack_json()
        self.threshold = threshold
        self._current: Optional[str] = None  # trace id of the grounding in progress, for demons
        self.frames = self._build_frames()
        self._tell_background(rules)

    # ----------------------------------------------------------------- frames

    def _build_frames(self) -> FrameSystem:
        fs = FrameSystem()
        fs.generic("asset")
        host = fs.generic("host", "asset")
        host.define("privilege_level", default="user", if_added=self._on_privilege)
        host.define("status", default="unknown")
        host.define("admin_sessions", if_needed=self._admin_sessions)
        host.define("trusted_by", if_needed=self._trusted_by)
        for name, parent in (("server", "host"), ("workstation", "host"), ("dmz", "host"),
                             ("database", "host"), ("domain_controller", "server")):
            fs.generic(name, parent)
        incident = fs.generic("incident")
        incident.define("tactic", if_needed=lambda f, s: self.net.tactic_of(f.get("technique")))
        incident.define("kill_chain_phase",
                        if_needed=lambda f, s: self.net.inherits(f.get("tactic"), "kill_chain_phase"))
        incident.define("mitigations", if_needed=lambda f, s: self.net.mitigations_for(f.get("technique")))
        for name, attrs in self.graph.nodes(data=True):
            kind = attrs.get("type", "workstation")
            if kind == "internet":
                continue
            fs.instance(name, kind if kind in fs else "host",
                        {"criticality": attrs.get("criticality", 0.0), "value": attrs.get("value", 0.0),
                         "admin_credentials": attrs.get("admin_credentials", False)})
        return fs

    def _domain_controllers(self) -> List[str]:
        return sorted(n for n, d in self.graph.nodes(data=True) if d.get("type") == "domain_controller")

    def _admin_sessions(self, frame: Frame, slot: str) -> List[str]:
        """Domain controllers this host holds cached admin credentials for."""
        if not frame.get("admin_credentials"):
            return []
        return [dc for dc in self._domain_controllers() if self.graph.has_edge(frame.name, dc)]

    def _trusted_by(self, frame: Frame, slot: str) -> List[str]:
        """Domain controllers that accept authentication from this host (it reaches them on the network)."""
        return [dc for dc in self._domain_controllers() if self.graph.has_edge(frame.name, dc)]

    def _on_privilege(self, frame: Frame, slot: str, value: str) -> None:
        """if_added demon: reaching SYSTEM on a host is privilege escalation."""
        if value != "SYSTEM":
            return
        tid = None
        if self.trace is not None:
            tid = self.trace.record("frame", f"{frame.name}.privilege_level = SYSTEM (if_added demon)",
                                    {"frame": frame.name, "slot": slot, "value": value}, [self._current])
        self.kb.tell(_a("PrivEsc", frame.name), trace_id=tid or self._current)

    # ----------------------------------------------------------------- background knowledge

    def topology_facts(self) -> Set[Atom]:
        facts: Set[Atom] = set()
        for f in self.frames.instances_of("host"):
            for dc in f.get("trusted_by") or []:
                facts.add(_a("Trusts", dc, f.name))
            for dc in f.get("admin_sessions") or []:
                facts.add(_a("HasAdminSession", f.name, dc))
        return facts

    def _tell_background(self, rules: Sequence[str]) -> None:
        tid = None
        if self.trace is not None:
            tid = self.trace.record("knowledge", "incident rules and topology facts",
                                    {"rules": list(rules), "hosts": self.graph.number_of_nodes()})
        for r in rules:
            self.kb.tell(r, trace_id=tid)
        for fact in sorted(self.topology_facts(), key=str):
            self.kb.tell(fact, trace_id=tid)

    # ----------------------------------------------------------------- grounding

    def posterior(self, alert: Alert, belief: Optional[BeliefState]) -> float:
        if belief is not None and alert.host in belief.host_compromise_prob:
            return belief.host_compromise_prob[alert.host]
        return alert.confidence

    def ground(self, alert: Alert, belief: Optional[BeliefState] = None) -> GroundedFacts:
        """Atoms for one alert, told to the knowledge base. Empty when below the threshold."""
        posterior = self.posterior(alert, belief)
        technique = self.net.technique_for_alert(alert.attack_class)
        parents = [alert.trace_id, belief.trace_id if belief else None]
        if posterior < self.threshold or technique is None or alert.host not in self.frames:
            reason = ("posterior below threshold" if posterior < self.threshold
                      else "unknown alert class" if technique is None else "unknown host")
            tid = self._record(f"skip {alert.attack_class} on {alert.host}: {reason}",
                               {"posterior": posterior, "reason": reason}, parents)
            return GroundedFacts(set(), {}, tid)

        tid = self._record(f"{alert.attack_class} on {alert.host} -> {technique} ({self.net.name(technique)})",
                           {"posterior": posterior, "technique": technique}, parents)
        self._current = tid
        try:
            incident = self.frames.instance(f"incident-{len(self.frames.instances_of('incident')) + 1}",
                                            "incident", {"host": alert.host, "technique": technique,
                                                         "alert_class": alert.attack_class,
                                                         "posterior": posterior})
            host = self.frames[alert.host]
            tactic = incident.get("tactic")

            atoms = {Atom("Uses", (Const(ATTACKER), Const(technique)))}
            if tactic in COMPROMISE_TACTICS:
                atoms.add(_a("Compromised", alert.host))
                host.set("status", "compromised")
            else:
                atoms.add(_a("Targeted", alert.host))
                host.set("status", "targeted")
            if self.net.is_a(technique, "T1003"):
                atoms.add(_a("CredentialsDumped", alert.host))

            for atom in sorted(atoms, key=str):
                self.kb.tell(atom, trace_id=tid)

            # A technique that needs higher privilege than the host frame records raises it (may fire a demon).
            required = self.net.inherits(technique, "requires_privilege", "user")
            if PRIVILEGE_RANK.get(required, 0) > PRIVILEGE_RANK.get(host.get("privilege_level"), 0):
                host.set("privilege_level", required)

            return GroundedFacts(atoms, {"incident": incident.to_dict(), "host": host.to_dict()}, tid)
        finally:
            self._current = None

    def _record(self, summary: str, payload: Dict, parents) -> Optional[str]:
        if self.trace is None:
            return None
        return self.trace.record("grounding", summary, payload, parents)


def ground(alert: Alert, belief: Optional[BeliefState], grounder: Grounder) -> Set[Atom]:
    """Functional form used in the design document: ground(alert, belief) -> set[Atom]."""
    return grounder.ground(alert, belief).atoms


def incident_demo(trace: Optional[TraceStore] = None) -> Dict[str, object]:
    """Credential dumping on Host3 leads, through the knowledge base, to a proof of Compromised(DC).

    Host3 holds a cached admin session to the domain controller. Nothing in the
    knowledge base says the DC is compromised until the alert is grounded.
    """
    from ..search.graph_builder import build_graph

    trace = trace if trace is not None else TraceStore()
    graph = build_graph(
        {
            "Internet": {"type": "internet"},
            "Host1": {"type": "workstation", "cves": [{"id": "CVE-SIM-1001", "cvss": 8.1}]},
            "Host2": {"type": "workstation"},
            "Host3": {"type": "workstation", "admin_credentials": True},
            "DC": {"type": "domain_controller", "criticality": 1.0, "value": 10.0},
        },
        [("Internet", "Host1"), ("Host1", "Host3"), ("Host1", "Host2"), ("Host2", "DC"), ("Host3", "DC")],
    )
    grounder = Grounder(graph, trace=trace)
    before = grounder.kb.ask("Compromised(DC)").proved

    ml = trace.record("ml", "credential_dumping on Host3 (p=0.93)", {"model": "decision tree"})
    hmm = trace.record("hmm", "phase = Exploitation (p=0.81)", {}, [ml])
    alert = Alert("Host3", "credential_dumping", 0.93, trace_id=ml)
    belief = BeliefState({"Exploitation": 0.81}, {"Host3": 0.9}, trace_id=hmm)
    grounded = grounder.ground(alert, belief)
    proof = grounder.kb.ask("Compromised(DC)")
    return {"graph": graph, "grounder": grounder, "trace": trace, "grounded": grounded,
            "proved_before": before, "proof": proof}

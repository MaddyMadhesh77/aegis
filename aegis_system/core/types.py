"""Records passed between pipeline stages.

Every record carries a `trace_id` pointing into the TraceStore, so any action the
system takes can be explained back to the detection that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .logic import Atom, Clause, Term, Var


@dataclass
class Alert:
    host: str
    attack_class: str
    confidence: float
    ttc_hours: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass
class BeliefState:
    phase_posterior: Dict[str, float]
    host_compromise_prob: Dict[str, float]
    trace_id: Optional[str] = None

    @property
    def max_host_prob(self) -> float:
        return max(self.host_compromise_prob.values(), default=0.0)


@dataclass
class GroundedFacts:
    atoms: Set[Atom]
    frames: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass
class ResolutionStep:
    """One clause in a resolution proof.

    Input clauses have no parents and `rule` set to "kb" or "negated_goal".
    Derived clauses list the ids of their parent clauses and the unifier used.
    """

    clause_id: int
    clause: Clause
    parents: Tuple[int, ...] = ()
    substitution: Dict[Var, Term] = field(default_factory=dict)
    rule: str = "kb"


@dataclass
class Proof:
    goal: Any
    steps: List[ResolutionStep]
    proved: bool
    bindings: Dict[Var, Term] = field(default_factory=dict)
    clauses_generated: int = 0
    trace_id: Optional[str] = None


@dataclass
class AttackPath:
    nodes: List[str]
    cost: float
    algorithm: str
    nodes_expanded: int = 0
    trace_id: Optional[str] = None


@dataclass
class DefenseObjective:
    goal_atoms: Set[Atom]
    choke_point: Any
    minimax_value: float
    trace_id: Optional[str] = None


@dataclass
class Plan:
    steps: Sequence[Any]
    orderings: Set[Tuple[Any, Any]] = field(default_factory=set)
    causal_links: Set[Tuple[Any, Atom, Any]] = field(default_factory=set)
    trace_id: Optional[str] = None


@dataclass
class ExecutionResult:
    step: Any
    success: bool
    observed_state: Dict[str, Any] = field(default_factory=dict)
    command: str = ""
    trace_id: Optional[str] = None

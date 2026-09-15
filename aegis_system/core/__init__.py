"""Shared foundations: logic representation, stage records, provenance and the simulated network."""

from .environment import ActionCommand, KILL_CHAIN, NetworkEnvironment, Observation
from .logic import (
    And, Atom, Clause, Const, Exists, Fn, ForAll, Iff, Implies, Literal, Not, Or, Var,
    parse, parse_atom, parse_term,
)
from .trace import TraceNode, TraceStore
from .types import (
    Alert, AttackPath, BeliefState, DefenseObjective, ExecutionResult, GroundedFacts, Plan, Proof,
    ResolutionStep,
)

__all__ = [
    "ActionCommand", "KILL_CHAIN", "NetworkEnvironment", "Observation",
    "And", "Atom", "Clause", "Const", "Exists", "Fn", "ForAll", "Iff", "Implies", "Literal", "Not", "Or", "Var",
    "parse", "parse_atom", "parse_term",
    "TraceNode", "TraceStore",
    "Alert", "AttackPath", "BeliefState", "DefenseObjective", "ExecutionResult", "GroundedFacts", "Plan",
    "Proof", "ResolutionStep",
]

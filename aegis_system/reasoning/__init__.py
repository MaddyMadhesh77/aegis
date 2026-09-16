"""Reasoning package: unification, clause form, resolution, chaining and DPLL."""

from .chaining import Rule, backward_chain, forward_chain
from .cnf import to_cnf
from .fol_engine import FOLKnowledgeBase, demo_kb
from .propositional import dpll, dpll_entails, tt_entails
from .resolution import format_proof, refute, resolve
from .unifier import compose, substitute, unify

__all__ = [
    "FOLKnowledgeBase", "Rule", "backward_chain", "compose", "demo_kb", "dpll", "dpll_entails",
    "format_proof", "forward_chain", "refute", "resolve", "substitute", "to_cnf", "tt_entails", "unify",
]

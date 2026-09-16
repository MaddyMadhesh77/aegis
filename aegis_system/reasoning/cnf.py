"""Conversion of first-order formulas to clause form (conjunctive normal form).

The pipeline, one function per step:

    1. eliminate_implications   a -> b  ==  ~a | b ;  a <-> b  ==  (~a | b) & (a | ~b)
    2. move_not_inwards         De Morgan, double negation, ~forall == exists ~
    3. standardize_variables    every quantifier gets its own variable
    4. skolemize                exists-variables become Skolem constants / functions
    5. drop_universals          remaining variables are implicitly universal
    6. distribute_or_over_and   (a & b) | c  ==  (a | c) & (b | c)

`to_cnf` runs all six and returns a list of clauses with tautologies removed.
Free variables in the input are treated as universally quantified.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

from ..core.logic import (
    And, Atom, Clause, Const, Exists, Fn, ForAll, Formula, Iff, Implies, Literal, Not, Or, Var, free_variables,
    fresh_var,
)
from .unifier import substitute

_skolem_counter = itertools.count(1)


def eliminate_implications(f: Formula) -> Formula:
    if isinstance(f, Atom):
        return f
    if isinstance(f, Not):
        return Not(eliminate_implications(f.arg))
    if isinstance(f, Implies):
        return Or(Not(eliminate_implications(f.left)), eliminate_implications(f.right))
    if isinstance(f, Iff):
        a, b = eliminate_implications(f.left), eliminate_implications(f.right)
        return And(Or(Not(a), b), Or(a, Not(b)))
    if isinstance(f, (And, Or)):
        return type(f)(eliminate_implications(f.left), eliminate_implications(f.right))
    if isinstance(f, (ForAll, Exists)):
        return type(f)(f.var, eliminate_implications(f.body))
    raise TypeError(f"Not a formula: {f!r}")


def move_not_inwards(f: Formula) -> Formula:
    """Negation normal form. Expects implications to be eliminated already."""
    if isinstance(f, Atom):
        return f
    if isinstance(f, Not):
        a = f.arg
        if isinstance(a, Atom):
            return f
        if isinstance(a, Not):
            return move_not_inwards(a.arg)
        if isinstance(a, And):
            return Or(move_not_inwards(Not(a.left)), move_not_inwards(Not(a.right)))
        if isinstance(a, Or):
            return And(move_not_inwards(Not(a.left)), move_not_inwards(Not(a.right)))
        if isinstance(a, ForAll):
            return Exists(a.var, move_not_inwards(Not(a.body)))
        if isinstance(a, Exists):
            return ForAll(a.var, move_not_inwards(Not(a.body)))
        raise TypeError(f"Eliminate implications before moving negations: {a!r}")
    if isinstance(f, (And, Or)):
        return type(f)(move_not_inwards(f.left), move_not_inwards(f.right))
    if isinstance(f, (ForAll, Exists)):
        return type(f)(f.var, move_not_inwards(f.body))
    raise TypeError(f"Eliminate implications before moving negations: {f!r}")


def standardize_variables(f: Formula) -> Formula:
    """Give every quantifier a fresh variable so no two quantifiers share a name."""

    def walk(node: Formula, mapping: Dict[Var, Var]) -> Formula:
        if isinstance(node, Atom):
            return substitute(node, mapping)
        if isinstance(node, Not):
            return Not(walk(node.arg, mapping))
        if isinstance(node, (And, Or, Implies, Iff)):
            return type(node)(walk(node.left, mapping), walk(node.right, mapping))
        if isinstance(node, (ForAll, Exists)):
            new_var = fresh_var(node.var)
            return type(node)(new_var, walk(node.body, {**mapping, node.var: new_var}))
        raise TypeError(f"Not a formula: {node!r}")

    return walk(f, {})


def skolemize(f: Formula, universals: Tuple[Var, ...] = ()) -> Formula:
    """Replace existential variables by Skolem terms over the enclosing universals.

    Expects negation normal form with standardized variables.
    """
    if isinstance(f, (Atom, Not)):
        return f
    if isinstance(f, (And, Or)):
        return type(f)(skolemize(f.left, universals), skolemize(f.right, universals))
    if isinstance(f, ForAll):
        return ForAll(f.var, skolemize(f.body, universals + (f.var,)))
    if isinstance(f, Exists):
        name = f"Sk{next(_skolem_counter)}"
        term = Fn(name, universals) if universals else Const(name)
        return skolemize(substitute(f.body, {f.var: term}), universals)
    raise TypeError(f"Convert to negation normal form before skolemizing: {f!r}")


def drop_universals(f: Formula) -> Formula:
    if isinstance(f, ForAll):
        return drop_universals(f.body)
    if isinstance(f, (And, Or)):
        return type(f)(drop_universals(f.left), drop_universals(f.right))
    if isinstance(f, Exists):
        raise TypeError("Skolemize before dropping universal quantifiers")
    return f


def distribute_or_over_and(f: Formula) -> Formula:
    if isinstance(f, And):
        return And(distribute_or_over_and(f.left), distribute_or_over_and(f.right))
    if isinstance(f, Or):
        a, b = distribute_or_over_and(f.left), distribute_or_over_and(f.right)
        if isinstance(a, And):
            return And(distribute_or_over_and(Or(a.left, b)), distribute_or_over_and(Or(a.right, b)))
        if isinstance(b, And):
            return And(distribute_or_over_and(Or(a, b.left)), distribute_or_over_and(Or(a, b.right)))
        return Or(a, b)
    return f


def _conjuncts(f: Formula) -> List[Formula]:
    if isinstance(f, And):
        return _conjuncts(f.left) + _conjuncts(f.right)
    return [f]


def _literals(f: Formula) -> List[Literal]:
    if isinstance(f, Or):
        return _literals(f.left) + _literals(f.right)
    if isinstance(f, Atom):
        return [Literal(f, True)]
    if isinstance(f, Not) and isinstance(f.arg, Atom):
        return [Literal(f.arg, False)]
    raise TypeError(f"Not a clause: {f}")


def is_tautology(clause: Clause) -> bool:
    return any(lit.negate() in clause for lit in clause)


def to_cnf_formula(f: Formula) -> Formula:
    """Run all six steps and return the CNF as a formula."""
    for v in reversed(free_variables(f)):  # free variables are implicitly universal
        f = ForAll(v, f)
    f = eliminate_implications(f)
    f = move_not_inwards(f)
    f = standardize_variables(f)
    f = skolemize(f)
    f = drop_universals(f)
    return distribute_or_over_and(f)


def to_cnf(f: Formula) -> List[Clause]:
    """Clause form of f, without tautologies or duplicate clauses, in a stable order."""
    clauses: List[Clause] = []
    for conjunct in _conjuncts(to_cnf_formula(f)):
        clause = frozenset(_literals(conjunct))
        if not is_tautology(clause) and clause not in clauses:
            clauses.append(clause)
    return clauses

"""Robinson's unification algorithm with occurs check.

A substitution maps Var -> Term. `unify` never modifies the substitution it is
given; it returns a new one, or None when the expressions cannot be unified.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..core.logic import (
    And, Atom, Const, Exists, Fn, ForAll, Iff, Implies, Literal, Not, Or, Term, Var,
)

Substitution = Dict[Var, Term]


def substitute(x: object, theta: Substitution) -> object:
    """Apply theta to a term, atom, literal, clause or formula, following variable chains."""
    if not theta:
        return x
    if isinstance(x, Var):
        if x in theta:
            return substitute(theta[x], theta)
        return x
    if isinstance(x, Const):
        return x
    if isinstance(x, Fn):
        return Fn(x.name, tuple(substitute(a, theta) for a in x.args))
    if isinstance(x, Atom):
        return Atom(x.pred, tuple(substitute(a, theta) for a in x.args))
    if isinstance(x, Literal):
        return Literal(substitute(x.atom, theta), x.positive)
    if isinstance(x, Not):
        return Not(substitute(x.arg, theta))
    if isinstance(x, (And, Or, Implies, Iff)):
        return type(x)(substitute(x.left, theta), substitute(x.right, theta))
    if isinstance(x, (ForAll, Exists)):
        inner = {v: t for v, t in theta.items() if v != x.var}  # bound variable is not substituted
        return type(x)(x.var, substitute(x.body, inner))
    if isinstance(x, frozenset):
        return frozenset(substitute(i, theta) for i in x)
    if isinstance(x, tuple):
        return tuple(substitute(i, theta) for i in x)
    if isinstance(x, list):
        return [substitute(i, theta) for i in x]
    return x


def occurs_in(var: Var, x: object, theta: Substitution) -> bool:
    """True if var appears in x once theta is applied (the occurs check)."""
    if x == var:
        return True
    if isinstance(x, Var):
        return x in theta and occurs_in(var, theta[x], theta)
    if isinstance(x, (Fn, Atom)):
        return any(occurs_in(var, a, theta) for a in x.args)
    if isinstance(x, (tuple, list)):
        return any(occurs_in(var, a, theta) for a in x)
    return False


def unify(x: object, y: object, theta: Optional[Substitution] = None) -> Optional[Substitution]:
    """Most general unifier of x and y extending theta, or None."""
    return _unify(x, y, dict(theta) if theta else {})


def _unify(x: object, y: object, theta: Optional[Substitution]) -> Optional[Substitution]:
    if theta is None:
        return None
    if x == y:
        return theta
    if isinstance(x, Var):
        return _unify_var(x, y, theta)
    if isinstance(y, Var):
        return _unify_var(y, x, theta)
    if isinstance(x, Fn) and isinstance(y, Fn):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        return _unify_sequence(x.args, y.args, theta)
    if isinstance(x, Atom) and isinstance(y, Atom):
        if x.pred != y.pred or len(x.args) != len(y.args):
            return None
        return _unify_sequence(x.args, y.args, theta)
    if isinstance(x, Literal) and isinstance(y, Literal):
        if x.positive != y.positive:
            return None
        return _unify(x.atom, y.atom, theta)
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        if len(x) != len(y):
            return None
        return _unify_sequence(x, y, theta)
    return None


def _unify_sequence(xs, ys, theta: Optional[Substitution]) -> Optional[Substitution]:
    for a, b in zip(xs, ys):
        theta = _unify(a, b, theta)
        if theta is None:
            return None
    return theta


def _unify_var(var: Var, x: object, theta: Substitution) -> Optional[Substitution]:
    if var in theta:
        return _unify(theta[var], x, theta)
    if isinstance(x, Var) and x in theta:
        return _unify(var, theta[x], theta)
    if occurs_in(var, x, theta):
        return None
    theta[var] = x
    return theta


def compose(theta1: Substitution, theta2: Substitution) -> Substitution:
    """Substitution equivalent to applying theta1 then theta2."""
    result: Substitution = {v: substitute(t, theta2) for v, t in theta1.items()}
    for v, t in theta2.items():
        result.setdefault(v, t)
    return {v: t for v, t in result.items() if v != t}


def resolve_bindings(theta: Substitution) -> Substitution:
    """Fully dereference every binding so the substitution is idempotent."""
    return {v: substitute(t, theta) for v, t in theta.items()}
